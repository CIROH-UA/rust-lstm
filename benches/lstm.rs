//! Benchmarks for the LSTM BMI.
//!
//! Two levels are measured, each on the Flex (production) and NdArray backends:
//!
//! * `forward_step` – one stateful forward pass of a single 10→64→1 `NextgenLstm`
//!   (batch 1, seq 1), i.e. the inner kernel ngen calls once per timestep per model.
//! * `bmi_update`   – a full `Bmi::update()` on a real ensemble config: setting the
//!   forcing inputs, scaling, running every ensemble member and averaging.
//!
//! The ensemble bench needs the trained models on disk. It looks in `DEFAULT_CONFIGS`,
//! or set `LSTM_BENCH_CONFIG=/path/to/config.yml` (relative `train_cfg_file` entries are
//! resolved against the config's directory). It is skipped if no config is found.
//!
//! Typical before/after comparison:
//!   cargo bench --bench lstm -- --save-baseline before
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench lstm -- --baseline before

use std::hint::black_box;
use std::path::{Path, PathBuf};

use bmi_rs::bmi::{Bmi, RefValues};
use burn::nn::LstmState;
use burn::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use rust_lstm_1025::LstmBmi;
use rust_lstm_1025::nextgen_lstm::NextgenLstm;

/// Searched in order when `LSTM_BENCH_CONFIG` is unset; `~` is `$HOME`.
const DEFAULT_CONFIGS: [&str; 2] = [
    "/ngen/ngen/extern/lstm/bmi_config_files/02046000_nh_AORC_hourly_ensemble.yml",
    "~/tools/lstm/bmi_config_files/02046000_nh_AORC_hourly_ensemble.yml",
];
const DEFAULT_WEIGHTS: &str = "/ngen/ngen/extern/lstm/trained_neuralhydrology_models/nh_AORC_hourly_25yr_1210_112435_7/model_epoch007.pt";

const INPUT_SIZE: usize = 10;
const HIDDEN_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 1;

/// Forcing values roughly representative of an hourly AORC timestep.
const FORCINGS: [(&str, f64); 8] = [
    ("atmosphere_water__liquid_equivalent_precipitation_rate", 0.8),
    ("land_surface_air__temperature", 285.0),
    ("land_surface_radiation~incoming~longwave__energy_flux", 320.0),
    ("land_surface_air__pressure", 100_500.0),
    ("atmosphere_air_water~vapor__relative_saturation", 0.007),
    ("land_surface_radiation~incoming~shortwave__energy_flux", 250.0),
    ("land_surface_wind__x_component_of_velocity", 1.5),
    ("land_surface_wind__y_component_of_velocity", -0.7),
];

fn forward_step(c: &mut Criterion, name: &str, device: Device) {
    let mut model = NextgenLstm::init(&device, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    if Path::new(DEFAULT_WEIGHTS).exists() {
        model.load_pytorch(DEFAULT_WEIGHTS).unwrap();
    } else {
        eprintln!("forward_step: {DEFAULT_WEIGHTS} not found, timing zero-initialised weights");
    }

    let input: Vec<f32> = (0..INPUT_SIZE).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut state: Option<LstmState<2>> = None;

    c.bench_function(&format!("forward_step/{name}"), |b| {
        b.iter(|| {
            let x = Tensor::<1>::from_floats(input.as_slice(), &device)
                .reshape([1, 1, INPUT_SIZE]);
            let (out, new_state) = model.forward(black_box(x), state.take());
            state = Some(new_state);
            // Pull the scalar back to the host, as the BMI does every step.
            let v: Vec<f32> = out.into_data().try_to_vec().unwrap();
            black_box(v[0])
        })
    });
}

fn bench_config() -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = match std::env::var("LSTM_BENCH_CONFIG") {
        Ok(p) => vec![p.into()],
        Err(_) => {
            let home = std::env::var("HOME").unwrap_or_default();
            DEFAULT_CONFIGS
                .iter()
                .map(|p| p.replacen('~', &home, 1).into())
                .collect()
        }
    };
    match candidates.iter().find(|p| p.exists()) {
        Some(p) => Some(p.canonicalize().unwrap()),
        None => {
            eprintln!("bmi_update: no config found (tried {candidates:?}), skipping");
            None
        }
    }
}

fn bmi_update(c: &mut Criterion, name: &str, device: Device, config: &Path) {
    // train_cfg_file entries are relative to the working directory.
    std::env::set_current_dir(config.parent().unwrap()).unwrap();

    let mut bmi = LstmBmi::new(device);
    bmi.initialize(config.to_str().unwrap())
        .expect("failed to initialize BMI");

    let mut step: u64 = 0;
    c.bench_function(&format!("bmi_update/{name}"), |b| {
        b.iter(|| {
            // Vary precip a little so nothing can be hoisted out of the loop.
            let wiggle = (step % 24) as f64 * 0.05;
            step += 1;
            for (var, value) in FORCINGS {
                let v = [value + wiggle];
                bmi.set_value(var, RefValues::F64(&v)).unwrap();
            }
            bmi.update().unwrap();
            black_box(bmi.get_value_ptr("land_surface_water__runoff_volume_flux").unwrap());
        })
    });

    bmi.finalize().unwrap();
}

fn benches(c: &mut Criterion) {
    #[allow(deprecated)] // NdArray is deprecated in burn 0.22; kept as a comparison point.
    let backends = [("flex", Device::flex()), ("ndarray", Device::ndarray())];

    for (name, device) in &backends {
        forward_step(c, name, device.clone());
    }
    if let Some(config) = bench_config() {
        for (name, device) in &backends {
            bmi_update(c, name, device.clone(), &config);
        }
    }
}

criterion_group!(lstm, benches);
criterion_main!(lstm);
