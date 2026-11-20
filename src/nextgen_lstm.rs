use std::path::Path;

use burn::nn::{Linear, LinearConfig, LstmState};
use burn::prelude::*;
use burn::store::{ModuleSnapshot, PytorchStore};
use burn::tensor::DType;
use burn::tensor::activation::{sigmoid, tanh};

/// Single-layer LSTM using PyTorch's fused-gate parameter layout.
///
/// `torch.nn.LSTM` stores all four gates stacked in one matrix per input
/// (`weight_ih_l0: [4*hidden, input]`, `weight_hh_l0: [4*hidden, hidden]`, plus matching
/// biases) in gate order input, forget, cell, output. Each `(weight, bias)` pair is exactly a
/// linear layer with `4*hidden` outputs, so it is modelled here as a [`Linear`]. That lets
/// [`PytorchStore`] load the checkpoint tensors as-is: its PyTorch adapter transposes `Linear`
/// weights automatically, and no per-gate splitting is needed.
#[derive(Module, Debug)]
pub struct FusedLstm {
    /// `x @ W_ih^T + b_ih` for all four gates.
    pub input_transform: Linear,
    /// `h @ W_hh^T + b_hh` for all four gates.
    pub hidden_transform: Linear,
    pub d_hidden: usize,
}

impl FusedLstm {
    pub fn init(device: &Device, d_input: usize, d_hidden: usize) -> Self {
        Self {
            input_transform: LinearConfig::new(d_input, 4 * d_hidden).init(device),
            hidden_transform: LinearConfig::new(d_hidden, 4 * d_hidden).init(device),
            d_hidden,
        }
    }

    /// Runs the LSTM over `input: [batch, seq, d_input]`, returning the hidden state at every
    /// timestep (`[batch, seq, d_hidden]`) and the final state. A `None` state starts from zeros.
    pub fn forward(
        &self,
        input: Tensor<3>,
        state: Option<LstmState<2>>,
    ) -> (Tensor<3>, LstmState<2>) {
        let device = input.device();
        let [batch_size, seq_length, _] = input.dims();
        let (mut cell, mut hidden) = state
            .unwrap_or_else(|| LstmState::initial([batch_size, self.d_hidden], &device))
            .unpack();

        // The input projection has no recurrence, so do it for the whole sequence at once.
        let input_gates = self.input_transform.forward(input);

        let mut outputs = Vec::with_capacity(seq_length);
        for t in 0..seq_length {
            let gates = input_gates.clone().slice(s![.., t]).squeeze_dim::<2>(1)
                + self.hidden_transform.forward(hidden);
            let [i, f, g, o]: [Tensor<2>; 4] = gates
                .chunk(4, 1)
                .try_into()
                .expect("gate dimension is 4 * d_hidden");

            cell = sigmoid(f) * cell + sigmoid(i) * tanh(g);
            hidden = sigmoid(o) * tanh(cell.clone());
            outputs.push(hidden.clone());
        }

        (Tensor::stack(outputs, 1), LstmState::new(cell, hidden))
    }
}

/// LSTM matching the PyTorch `Nextgen_CudaLSTM` / neuralhydrology `CudaLSTM` architecture:
/// one LSTM layer followed by a linear regression head.
#[derive(Module, Debug)]
pub struct NextgenLstm {
    pub lstm: FusedLstm,
    pub head: Linear,
}

impl NextgenLstm {
    pub fn init(
        device: &Device,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> NextgenLstm {
        Self {
            lstm: FusedLstm::init(device, input_size, hidden_size),
            head: LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    /// Loads a neuralhydrology PyTorch checkpoint (e.g. `model_epoch007.pt`) directly.
    ///
    /// Every tensor in the module must be present in the file with the expected shape and every
    /// tensor in the file must be used, otherwise an error is returned.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        let mut store = PytorchStore::from_file(path)
            // lstm.{weight,bias}_ih_l0 -> lstm.input_transform.{weight,bias}
            .with_key_remapping(r"^lstm\.(weight|bias)_ih_l0$", "lstm.input_transform.$1")
            // lstm.{weight,bias}_hh_l0 -> lstm.hidden_transform.{weight,bias}
            .with_key_remapping(r"^lstm\.(weight|bias)_hh_l0$", "lstm.hidden_transform.$1")
            // head.net.0.{weight,bias} -> head.{weight,bias}
            .with_key_remapping(r"^head\.net\.0\.", "head.");

        let result = self.load_from(&mut store)?;
        if !result.is_success() || !result.missing.is_empty() || !result.unused.is_empty() {
            return Err(format!(
                "weights in {} do not match the model: {result}",
                path.display()
            )
            .into());
        }

        // burn-store keeps the checkpoint's float dtype; the BMI feeds f32 inputs, so make
        // sure f64/f16 checkpoints don't end up with mismatched parameter dtypes.
        self.lstm.input_transform = linear_to_f32(self.lstm.input_transform.clone());
        self.lstm.hidden_transform = linear_to_f32(self.lstm.hidden_transform.clone());
        self.head = linear_to_f32(self.head.clone());
        Ok(())
    }

    pub fn forward(
        &self,
        input: Tensor<3>,
        state: Option<LstmState<2>>,
    ) -> (Tensor<3>, LstmState<2>) {
        let (output, state) = self.lstm.forward(input, state);
        (self.head.forward(output), state)
    }
}

fn linear_to_f32(linear: Linear) -> Linear {
    Linear {
        weight: linear.weight.map(|w| w.cast(DType::F32)),
        bias: linear.bias.map(|b| b.map(|b| b.cast(DType::F32))),
    }
}
