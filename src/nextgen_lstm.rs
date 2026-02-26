use crate::weights::WeightConfig;
use burn::{
    module::Param,
    nn::{
        GateController, Initializer, Linear, LinearConfig, LinearRecord, Lstm, LstmConfig,
        LstmState,
    },
    prelude::*,
    tensor::Bytes,
};
use serde_json::Value;
use std::fs;
/// Simple LSTM matching the PyTorch Nextgen_CudaLSTM architecture
#[derive(Module, Debug)]
pub struct NextgenLstm<B: Backend> {
    pub lstm: Lstm<B>,
    pub head: Linear<B>,
}

pub fn create_with_weights<B: Backend>(
    d_input: usize,
    d_output: usize,
    bias: bool,
    initializer: Initializer,
    input_record: crate::nn::LinearRecord<B>,
    hidden_record: crate::nn::LinearRecord<B>,
) -> GateController<B> {
    let l1 = LinearConfig {
        d_input,
        d_output,
        bias,
        initializer: initializer.clone(),
    }
    .init(&input_record.weight.device())
    .load_record(input_record);
    let l2 = LinearConfig {
        d_input,
        d_output,
        bias,
        initializer,
    }
    .init(&hidden_record.weight.device())
    .load_record(hidden_record);

    GateController {
        input_transform: l1,
        hidden_transform: l2,
    }
}

pub fn vec_to_tensor(input_vec: &Vec<f32>, shape: Vec<usize>) -> TensorData {
    let bytes_vec = input_vec
        .iter()
        .flat_map(|&value| f32::to_le_bytes(value))
        .collect();
    let bytes = Bytes::from_bytes_vec(bytes_vec);
    TensorData {
        bytes,
        shape,
        dtype: burn::tensor::DType::F32,
    }
}

pub fn serde_to_vec(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn create_gate_controller<B: Backend>(
    input_weights: &Vec<f32>,
    input_biases: &Vec<f32>,
    hidden_weights: &Vec<f32>,
    hidden_biases: &Vec<f32>,
    device: &Device<B>,
    input_length: usize,
    hidden_size: usize,
) -> GateController<B> {
    let input_record = LinearRecord {
        weight: Param::from_data(
            vec_to_tensor(input_weights, vec![input_length, hidden_size]),
            device,
        ),
        bias: Some(Param::from_data(
            vec_to_tensor(input_biases, vec![hidden_size]),
            device,
        )),
    };
    let hidden_record = LinearRecord {
        weight: Param::from_data(
            vec_to_tensor(hidden_weights, vec![hidden_size, hidden_size]),
            device,
        ),
        bias: Some(Param::from_data(
            vec_to_tensor(hidden_biases, vec![hidden_size]),
            device,
        )),
    };

    create_with_weights(
        input_length,
        hidden_size,
        true,
        Initializer::Zeros,
        input_record,
        hidden_record,
    )
}

impl<B: Backend> NextgenLstm<B> {
    /// Forward pass matching PyTorch implementation
    pub fn init(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> NextgenLstm<B> {
        let lstm = LstmConfig::new(input_size, hidden_size, true)
            .with_initializer(nn::Initializer::Zeros)
            .init(device);
        let head = LinearConfig::new(hidden_size, output_size)
            .with_bias(true)
            .init(device);
        Self { lstm, head }
    }
    fn get_json_weights(weight_path: &str) -> Option<WeightConfig> {
        let json_str = fs::read_to_string(weight_path);
        if let Err(e) = json_str {
            eprintln!("Error reading weights file at ({}): {}", weight_path, e);
            return None;
        }
        let weights: Value = serde_json::from_str(&json_str.unwrap()).unwrap();
        return Some(WeightConfig {
            input_size: weights["input_size"].as_u64().unwrap() as usize,
            hidden_size: weights["hidden_size"].as_u64().unwrap() as usize,
            output_size: weights["output_size"].as_u64().unwrap() as usize,
            input_gate_input_weights: serde_to_vec(&weights["lstm.input_gate.input_transform.weight"]),
            input_gate_input_biases: serde_to_vec(&weights["lstm.input_gate.input_transform.bias"]),
            input_gate_hidden_weights: serde_to_vec(&weights["lstm.input_gate.hidden_transform.weight"]),
            input_gate_hidden_biases: serde_to_vec(&weights["lstm.input_gate.hidden_transform.bias"]),
            forget_gate_input_weights: serde_to_vec(&weights["lstm.forget_gate.input_transform.weight"]),
            forget_gate_input_biases: serde_to_vec(&weights["lstm.forget_gate.input_transform.bias"]),
            forget_gate_hidden_weights: serde_to_vec(&weights["lstm.forget_gate.hidden_transform.weight"]),
            forget_gate_hidden_biases: serde_to_vec(&weights["lstm.forget_gate.hidden_transform.bias"]),
            cell_gate_input_weights: serde_to_vec(&weights["lstm.cell_gate.input_transform.weight"]),
            cell_gate_input_biases: serde_to_vec(&weights["lstm.cell_gate.input_transform.bias"]),
            cell_gate_hidden_weights: serde_to_vec(&weights["lstm.cell_gate.hidden_transform.weight"]),
            cell_gate_hidden_biases: serde_to_vec(&weights["lstm.cell_gate.hidden_transform.bias"]),
            output_gate_input_weights: serde_to_vec(&weights["lstm.output_gate.input_transform.weight"]),
            output_gate_input_biases: serde_to_vec(&weights["lstm.output_gate.input_transform.bias"]),
            output_gate_hidden_weights: serde_to_vec(&weights["lstm.output_gate.hidden_transform.weight"]),
            output_gate_hidden_biases: serde_to_vec(&weights["lstm.output_gate.hidden_transform.bias"]),
        });
    }
    fn get_hardcoded_weights(weight_path: &str) -> Option<WeightConfig> {
        let weight_dirname: &str = weight_path.split('/').rev().nth(2).unwrap();
        match weight_dirname {
            "nh_AORC_hourly_25yr_1210_112435_7" => WeightConfig::nh_AORC_hourly_25yr_1210_112435_7(),
            "nh_AORC_hourly_25yr_1210_112435_8" => WeightConfig::nh_AORC_hourly_25yr_1210_112435_8(),
            "nh_AORC_hourly_25yr_1210_112435_9" => WeightConfig::nh_AORC_hourly_25yr_1210_112435_9(),
            "nh_AORC_hourly_25yr_seq999_seed101_0701_143442" => WeightConfig::nh_AORC_hourly_25yr_seq999_seed101_0701_143442(),
            "nh_AORC_hourly_25yr_seq999_seed103_2701_171540" => WeightConfig::nh_AORC_hourly_25yr_seq999_seed103_2701_171540(),
            "nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806" => WeightConfig::nh_AORC_hourly_slope_elev_precip_temp_seq999_seed101_2801_191806(),
            _ => None,
        }
    }   
    pub fn load_weightconfig(&mut self, device: &B::Device, weight_config: WeightConfig) {
        self.lstm.input_gate = create_gate_controller(
            &weight_config.input_gate_input_weights,
            &weight_config.input_gate_input_biases,
            &weight_config.input_gate_hidden_weights,
            &weight_config.input_gate_hidden_biases,
            device,
            weight_config.input_size,
            weight_config.hidden_size,
        );
        self.lstm.forget_gate = create_gate_controller(
            &weight_config.forget_gate_input_weights,
            &weight_config.forget_gate_input_biases,
            &weight_config.forget_gate_hidden_weights,
            &weight_config.forget_gate_hidden_biases,
            device,
            weight_config.input_size,
            weight_config.hidden_size,
        );
        self.lstm.cell_gate = create_gate_controller(
            &weight_config.cell_gate_input_weights,
            &weight_config.cell_gate_input_biases,
            &weight_config.cell_gate_hidden_weights,
            &weight_config.cell_gate_hidden_biases,
            device,
            weight_config.input_size,
            weight_config.hidden_size,
        );
        self.lstm.output_gate = create_gate_controller(
            &weight_config.output_gate_input_weights,
            &weight_config.output_gate_input_biases,
            &weight_config.output_gate_hidden_weights,
            &weight_config.output_gate_hidden_biases,
            device,
            weight_config.input_size,
            weight_config.hidden_size,
        );
    }
    pub fn load_json_weights(&mut self, device: &B::Device, weight_path: &str) {
        let weight_config = Self::get_json_weights(weight_path);
        if weight_config.is_none() {
            panic!("Failed to load weights from path: {}", weight_path);
        } else {
            self.load_weightconfig(device, weight_config.unwrap());
        }
    }
    pub fn load_weights(&mut self, device: &B::Device, weight_path: &str) {
        // Try to load hardcoded weights first, then fall back to json loading if that fails
        if let Some(weight_config) = Self::get_hardcoded_weights(weight_path) {
            self.load_weightconfig(device, weight_config);
        } else {
            self.load_json_weights(device, weight_path);
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 3>, LstmState<B, 2>) {
        // dbg!(&input);
        let (output, state) = self.lstm.forward(input, state);
        // Apply head to each timestep
        let [batch_size, seq_length, hidden_size] = output.dims();
        // dbg!(&output);
        let output_reshaped = output.reshape([batch_size * seq_length, hidden_size]);
        let prediction = self.head.forward(output_reshaped);
        // dbg!(&prediction);
        let prediction = prediction.reshape([batch_size, seq_length, 1]);
        (prediction, state)
    }
}
