use bmi_rs::bmi::{Bmi, BmiResult, Location, RefValues, ValueType, Values, register_model};
use burn::prelude::*;
use glob::glob;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[doc(hidden)]
pub mod ensemble;
#[doc(hidden)]
pub mod nextgen_lstm;
pub use ensemble::EnsembleLayout;
use ensemble::StackedEnsemble;
use nextgen_lstm::NextgenLstm;

/// Environment variable overriding how ensemble members are stacked (see [`EnsembleLayout`]).
pub const ENSEMBLE_LAYOUT_ENV: &str = "RUST_LSTM_ENSEMBLE_LAYOUT";

/// The fields we need from a neuralhydrology training `config.yml`.
#[derive(Debug, Deserialize)]
struct TrainingConfig {
    run_dir: PathBuf,
    hidden_size: usize,
    dynamic_inputs: Vec<String>,
    #[serde(default)]
    static_attributes: Option<Vec<String>>,
    target_variables: Vec<String>,
}

/// neuralhydrology's `train_data/train_data_scaler.yml`. The dynamic-feature scalers are
/// serialised xarray datasets (`xarray.Dataset.to_dict()`), the static ones are plain maps.
#[derive(Debug, Deserialize)]
struct ScalerFile {
    #[serde(default)]
    attribute_means: Option<HashMap<String, f64>>,
    #[serde(default)]
    attribute_stds: Option<HashMap<String, f64>>,
    xarray_feature_center: XarrayDataset,
    xarray_feature_scale: XarrayDataset,
}

#[derive(Debug, Deserialize)]
struct XarrayDataset {
    data_vars: HashMap<String, XarrayVariable>,
}

#[derive(Debug, Deserialize)]
struct XarrayVariable {
    data: f64,
}

#[derive(Debug)]
struct TrainingScalars {
    input_mean: Vec<f32>,
    input_std: Vec<f32>,
    output_mean: f32,
    output_std: f32,
}

impl TrainingScalars {
    /// Builds the input scalers in model input order (dynamic inputs, then static attributes)
    /// and the output scaler for the first target variable.
    fn from_scaler_file(scaler: &ScalerFile, config: &TrainingConfig) -> BmiResult<Self> {
        let dynamic = |dataset: &XarrayDataset, name: &str| {
            dataset
                .data_vars
                .get(name)
                .map(|v| v.data as f32)
                .ok_or_else(|| format!("train_data_scaler.yml has no scaler for {name}"))
        };
        let attribute = |map: &Option<HashMap<String, f64>>, name: &str| {
            map.as_ref()
                .and_then(|m| m.get(name))
                .map(|v| *v as f32)
                .ok_or_else(|| format!("train_data_scaler.yml has no scaler for {name}"))
        };

        let mut input_mean = Vec::new();
        let mut input_std = Vec::new();
        for name in &config.dynamic_inputs {
            input_mean.push(dynamic(&scaler.xarray_feature_center, name)?);
            input_std.push(dynamic(&scaler.xarray_feature_scale, name)?);
        }
        for name in config.static_attributes.iter().flatten() {
            input_mean.push(attribute(&scaler.attribute_means, name)?);
            input_std.push(attribute(&scaler.attribute_stds, name)?);
        }

        let target = config
            .target_variables
            .first()
            .ok_or("training config has no target_variables")?;
        Ok(Self {
            input_mean,
            input_std,
            output_mean: dynamic(&scaler.xarray_feature_center, target)?,
            output_std: dynamic(&scaler.xarray_feature_scale, target)?,
        })
    }
}

/// Removes `..` components textually (e.g. `../models/../models/x` -> `../models/x`) so a
/// relative `run_dir` isn't resolved through symlinked directories by the OS.
fn normalize_lexically(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out: Vec<Component> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir if matches!(out.last(), Some(Component::Normal(_))) => {
                out.pop();
            }
            Component::ParentDir if matches!(out.last(), Some(Component::RootDir)) => {}
            other => out.push(other),
        }
    }
    out.iter().collect()
}

// Per-member metadata; the weights themselves live in the stacked ensemble.
struct ModelInstance {
    input_names: Vec<String>,
    scalars: TrainingScalars,
}

// Macro to handle variable lookup
macro_rules! match_var {
    ($name:expr, $($pattern:pat => $result:expr),+ $(,)?) => {
        match $name {
            $($pattern => $result,)+
            _ => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", $name)
            )))
        }
    };
}

pub struct LstmBmi {
    // Ensemble of models
    models: Vec<ModelInstance>,
    ensemble: Option<StackedEnsemble>,
    layout: EnsembleLayout,
    device: Device,

    // Configuration
    config_path: String,
    area_sqkm: f32,
    output_scale_factor_cms: f32,

    // BMI variables stored as simple HashMap
    variables: HashMap<String, Vec<f64>>,

    // Variable metadata
    input_var_names: Vec<&'static str>,
    output_var_names: Vec<&'static str>,

    // Time tracking
    current_time: f64,
    start_time: f64,
    end_time: f64,
    time_step: f64,
}

impl LstmBmi {
    pub fn new(device: Device) -> Self {
        // Define all BMI variable names
        let input_vars = vec![
            "atmosphere_water__liquid_equivalent_precipitation_rate",
            "land_surface_air__temperature",
            "land_surface_radiation~incoming~longwave__energy_flux",
            "land_surface_air__pressure",
            "atmosphere_air_water~vapor__relative_saturation",
            "land_surface_radiation~incoming~shortwave__energy_flux",
            "land_surface_wind__x_component_of_velocity",
            "land_surface_wind__y_component_of_velocity",
            // "basin__mean_of_elevation",
            // "basin__mean_of_slope",
        ];

        let output_vars = vec![
            "land_surface_water__runoff_volume_flux",
            "land_surface_water__runoff_depth",
        ];

        // Initialize variables HashMap
        let mut variables = HashMap::new();
        for var in &input_vars {
            variables.insert(var.to_string(), vec![0.0]);
        }
        // add variables here so bmi can't see them
        variables.insert("basin__mean_of_elevation".to_string(), vec![0.0]);
        variables.insert("basin__mean_of_slope".to_string(), vec![0.0]);

        for var in &output_vars {
            variables.insert(var.to_string(), vec![0.0]);
        }

        LstmBmi {
            models: Vec::new(),
            ensemble: None,
            layout: EnsembleLayout::BatchedByHidden,
            device,
            config_path: String::new(),
            area_sqkm: 0.0,
            output_scale_factor_cms: 0.0,
            variables,
            input_var_names: input_vars,
            output_var_names: output_vars,
            current_time: 0.0,
            start_time: 0.0,
            end_time: 36000.0, // 10 hours default
            time_step: 3600.0, // 1 hour
        }
    }

    fn internal_to_external_name(&self, internal: &str) -> String {
        let mapping = [
            (
                "DLWRF_surface",
                "land_surface_radiation~incoming~longwave__energy_flux",
            ),
            ("PRES_surface", "land_surface_air__pressure"),
            (
                "SPFH_2maboveground",
                "atmosphere_air_water~vapor__relative_saturation",
            ),
            (
                "APCP_surface",
                "atmosphere_water__liquid_equivalent_precipitation_rate",
            ),
            (
                "DSWRF_surface",
                "land_surface_radiation~incoming~shortwave__energy_flux",
            ),
            ("TMP_2maboveground", "land_surface_air__temperature"),
            (
                "UGRD_10maboveground",
                "land_surface_wind__x_component_of_velocity",
            ),
            (
                "VGRD_10maboveground",
                "land_surface_wind__y_component_of_velocity",
            ),
            ("elev_mean", "basin__mean_of_elevation"),
            ("slope_mean", "basin__mean_of_slope"),
        ];

        mapping
            .iter()
            .find(|(k, _)| *k == internal)
            .map(|(_, v)| v.to_string())
            .unwrap_or_else(|| internal.to_string())
    }

    /// Uses `layout` to stack the ensemble members, unless overridden by the
    /// [`ENSEMBLE_LAYOUT_ENV`] environment variable.
    pub fn with_layout(mut self, layout: EnsembleLayout) -> Self {
        self.layout = layout;
        self
    }

    fn load_single_model(
        &self,
        training_config_path: &Path,
    ) -> BmiResult<(NextgenLstm, ModelInstance)> {
        let config: TrainingConfig =
            serde_yaml::from_str(&fs::read_to_string(training_config_path)?)?;

        // A relative run_dir is relative to the directory containing the run directories,
        // i.e. the grandparent of `<runs>/<run>/config.yml`.
        let run_dir = if config.run_dir.is_absolute() {
            config.run_dir.clone()
        } else {
            normalize_lexically(
                &training_config_path
                    .parent()
                    .and_then(Path::parent)
                    .unwrap_or(Path::new(""))
                    .join(&config.run_dir),
            )
        };

        let model_path = glob(&format!("{}/model_*.pt", run_dir.display()))?
            .next()
            .ok_or_else(|| format!("No model_*.pt file found in {}", run_dir.display()))??;

        let scaler_path = run_dir.join("train_data").join("train_data_scaler.yml");
        let scaler: ScalerFile = serde_yaml::from_str(&fs::read_to_string(&scaler_path)?)?;
        let scalars = TrainingScalars::from_scaler_file(&scaler, &config)?;

        let input_names: Vec<String> = config
            .dynamic_inputs
            .iter()
            .chain(config.static_attributes.iter().flatten())
            .cloned()
            .collect();

        let mut model = NextgenLstm::init(
            &self.device,
            input_names.len(),
            config.hidden_size,
            config.target_variables.len(),
        );
        model
            .load_pytorch(&model_path)
            .map_err(|e| format!("failed to load weights from {}: {e}", model_path.display()))?;

        Ok((
            model,
            ModelInstance {
                input_names,
                scalars,
            },
        ))
    }

    fn run_ensemble(&mut self) -> BmiResult<()> {
        if self.models.is_empty() {
            return Err("No models in ensemble")?;
        }
        // Gather and scale each member's inputs, in that member's input order
        let scaled_inputs: Vec<Vec<f32>> = self
            .models
            .iter()
            .map(|model| {
                model
                    .input_names
                    .iter()
                    .zip(&model.scalars.input_mean)
                    .zip(&model.scalars.input_std)
                    .map(|((name, mean), std)| {
                        let value = self
                            .variables
                            .get(&self.internal_to_external_name(name))
                            .and_then(|v| v.first())
                            .copied()
                            .unwrap_or(0.0) as f32;
                        if *std != 0.0 {
                            (value - mean) / std
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // One timestep of every member, then denormalise each member's output
        let raw_outputs = self
            .ensemble
            .as_mut()
            .ok_or("Ensemble not initialized")?
            .step(&scaled_inputs, &self.device);
        let ensemble_outputs: Vec<f32> = raw_outputs
            .iter()
            .zip(&self.models)
            .map(|(output, model)| {
                (output * model.scalars.output_std + model.scalars.output_mean).max(0.0)
            })
            .collect();

        // Calculate mean of ensemble outputs
        let mean_surface_runoff_mm = if !ensemble_outputs.is_empty() {
            ensemble_outputs.iter().sum::<f32>() / ensemble_outputs.len() as f32
        } else {
            0.0
        };

        // Convert to output units
        let surface_runoff_m = mean_surface_runoff_mm / 1000.0;
        let surface_runoff_volume_m3_s = mean_surface_runoff_mm * self.output_scale_factor_cms;

        // Set outputs
        self.variables.insert(
            "land_surface_water__runoff_depth".to_string(),
            vec![surface_runoff_m as f64],
        );
        self.variables.insert(
            "land_surface_water__runoff_volume_flux".to_string(),
            vec![surface_runoff_volume_m3_s as f64],
        );
        Ok(())
    }
}

impl Bmi for LstmBmi {
    fn initialize(&mut self, config_file: &str) -> BmiResult<()> {
        // println!("Initializing LSTM BMI with config: {}", config_file);
        self.config_path = config_file.to_string();

        // Load configuration
        let config_path = Path::new(config_file);
        let config_str = fs::read_to_string(config_path)?;
        let config: serde_yaml::Value = serde_yaml::from_str(&config_str)?;

        // Get all training config paths (ensemble)
        let training_configs = config["train_cfg_file"]
            .as_sequence()
            .ok_or("train_cfg_file should be an array")?;

        // println!("Loading ensemble of {} models", training_configs.len());

        // Load each model in the ensemble
        let mut networks = Vec::with_capacity(training_configs.len());
        for (idx, config_value) in training_configs.iter().enumerate() {
            let training_config_path = Path::new(
                config_value
                    .as_str()
                    .ok_or(format!("train_cfg_file[{}] not a string", idx))?,
            );

            // println!(
            //     "Loading model {}/{}: {}",
            //     idx + 1,
            //     training_configs.len(),
            //     training_config_path.display()
            // );

            let (network, model_instance) = self.load_single_model(training_config_path)?;
            networks.push(network);
            self.models.push(model_instance);
        }

        // Fuse the members into stacked LSTMs
        let layout = match std::env::var(ENSEMBLE_LAYOUT_ENV) {
            Ok(name) => name.parse::<EnsembleLayout>()?,
            Err(_) => self.layout,
        };
        let networks: Vec<&NextgenLstm> = networks.iter().collect();
        self.ensemble = Some(StackedEnsemble::new(&networks, layout, &self.device));

        // Get area from config
        self.area_sqkm = config
            .get("area_sqkm")
            .ok_or("Missing area_sqkm")?
            .as_f64()
            .ok_or("area_sqkm not a number")? as f32;

        self.output_scale_factor_cms =
            (1.0 / 1000.0) * (self.area_sqkm * 1000.0 * 1000.0) * (1.0 / 3600.0);

        // Set static inputs from config
        let elevation = config
            .get("elev_mean")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let slope = config
            .get("slope_mean")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        self.variables
            .insert("basin__mean_of_elevation".to_string(), vec![elevation]);
        self.variables
            .insert("basin__mean_of_slope".to_string(), vec![slope]);

        // Reset time
        self.current_time = self.start_time;

        // println!(
        //     "LSTM BMI ensemble initialized successfully with {} models",
        //     self.models.len()
        // );
        Ok(())
    }

    fn update(&mut self) -> BmiResult<()> {
        self.run_ensemble()?;
        self.current_time += self.time_step;
        Ok(())
    }

    fn update_until(&mut self, then: f64) -> BmiResult<()> {
        if then < self.current_time {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Target time {} is before current time {}",
                    then, self.current_time
                ),
            )));
        }

        while self.current_time < then {
            self.update()?;
            if self.current_time > then {
                self.current_time = then;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> BmiResult<()> {
        self.models.clear();
        self.ensemble = None;
        Ok(())
    }

    fn get_component_name(&self) -> &str {
        "NextGen LSTM BMI Ensemble"
    }

    fn get_input_item_count(&self) -> u32 {
        self.input_var_names.len() as u32
    }

    fn get_output_item_count(&self) -> u32 {
        self.output_var_names.len() as u32
    }

    fn get_input_var_names(&self) -> &[&str] {
        &self.input_var_names
    }

    fn get_output_var_names(&self) -> &[&str] {
        &self.output_var_names
    }

    fn get_var_grid(&self, name: &str) -> BmiResult<i32> {
        if self.variables.contains_key(name) {
            Ok(0) // Scalar grid
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )))
        }
    }

    fn get_var_type(&self, name: &str) -> BmiResult<ValueType> {
        if self.variables.contains_key(name) {
            Ok(ValueType::F64)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )))
        }
    }

    fn get_var_units(&self, name: &str) -> BmiResult<&str> {
        match_var!(name,
            "atmosphere_water__liquid_equivalent_precipitation_rate" => Ok("mm h-1"),
            "land_surface_air__temperature" => Ok("degK"),
            "land_surface_radiation~incoming~longwave__energy_flux" => Ok("W m-2"),
            "land_surface_air__pressure" => Ok("Pa"),
            "atmosphere_air_water~vapor__relative_saturation" => Ok("kg kg-1"),
            "land_surface_radiation~incoming~shortwave__energy_flux" => Ok("W m-2"),
            "land_surface_wind__x_component_of_velocity" => Ok("m s-1"),
            "land_surface_wind__y_component_of_velocity" => Ok("m s-1"),
            "basin__mean_of_elevation" => Ok("m"),
            "basin__mean_of_slope" => Ok("m km-1"),
            "land_surface_water__runoff_volume_flux" => Ok("m3 s-1"),
            "land_surface_water__runoff_depth" => Ok("m")
        )
    }

    fn get_var_nbytes(&self, name: &str) -> BmiResult<u32> {
        let itemsize = self.get_var_itemsize(name)?;
        let values = self.get_value_ptr(name)?;
        Ok(values.len() as u32 * itemsize)
    }

    fn get_var_location(&self, name: &str) -> BmiResult<Location> {
        if self.variables.contains_key(name) {
            Ok(Location::Node)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )))
        }
    }

    fn get_current_time(&self) -> f64 {
        self.current_time
    }

    fn get_start_time(&self) -> f64 {
        self.start_time
    }

    fn get_end_time(&self) -> f64 {
        self.end_time
    }

    fn get_time_units(&self) -> &str {
        "seconds"
    }

    fn get_time_step(&self) -> f64 {
        self.time_step
    }

    fn get_value_ptr(&self, name: &str) -> BmiResult<RefValues<'_>> {
        Ok(self
            .variables
            .get(name)
            .map(|v| RefValues::F64(v))
            .ok_or_else(|| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", name),
                ))
            })?)
    }

    fn get_value_at_indices(&self, name: &str, inds: &[u32]) -> BmiResult<Values> {
        let values = self.variables.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Variable {} not found", name),
            )
        })?;

        let mut result = Vec::with_capacity(inds.len());
        for &idx in inds {
            if (idx as usize) >= values.len() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Index {} out of bounds", idx),
                )));
            }
            result.push(values[idx as usize]);
        }
        Ok(Values::F64(result))
    }

    fn set_value(&mut self, name: &str, src: RefValues) -> BmiResult<()> {
        if let RefValues::F64(values) = src {
            if let Some(var) = self.variables.get_mut(name) {
                if values.len() != var.len() {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Source array size mismatch",
                    )));
                }
                var.copy_from_slice(values);
                Ok(())
            } else {
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", name),
                )))
            }
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Type mismatch: expected F64",
            )))
        }
    }

    fn set_value_at_indices(&mut self, name: &str, inds: &[u32], src: RefValues) -> BmiResult<()> {
        if let RefValues::F64(values) = src {
            if values.len() != inds.len() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Source array size doesn't match indices count",
                )));
            }

            let var = self.variables.get_mut(name).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Variable {} not found", name),
                )
            })?;

            for (i, &idx) in inds.iter().enumerate() {
                if (idx as usize) >= var.len() {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Index {} out of bounds", idx),
                    )));
                }
                var[idx as usize] = values[i];
            }
            Ok(())
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Type mismatch: expected F64",
            )))
        }
    }
}

// Export function for C binding
#[unsafe(no_mangle)]
pub extern "C" fn register_bmi_lstm(handle: *mut ffi::Bmi) -> *mut ffi::Bmi {
    let model = LstmBmi::new(Device::flex());
    register_model(handle, model);
    handle
}

#[cfg(test)]
mod tests {
    use super::normalize_lexically;
    use std::path::{Path, PathBuf};

    #[test]
    fn normalizes_relative_run_dirs() {
        let cases = [
            ("../models/../models/run", "../models/run"),
            ("../../a/b", "../../a/b"),
            ("/x/models/../y", "/x/y"),
            ("/../y", "/y"),
            ("models/./run", "models/run"),
            ("run", "run"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_lexically(Path::new(input)),
                PathBuf::from(expected),
                "{input}"
            );
        }
    }
}
