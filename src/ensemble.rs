//! Ensemble members fused into stacked LSTMs, so one step of the whole ensemble is a handful of
//! tensor ops instead of a full forward pass per member.
//!
//! Every layout uses the same representation: a [`Stack`] holds parameters for `B` independent
//! "lanes" (`w_ih: [B, I, 4H]`, `w_hh: [B, H, 4H]`, `head_w: [B, H, O]`) and runs them with
//! batched matmuls. A [`Slot`] records where one member's weights sit inside a stack:
//!
//! * [`EnsembleLayout::BlockDiagonal`]: one lane holding a single wide LSTM. Members are placed
//!   side by side, so `W_ih` and `W_hh` are block diagonal per gate (members have different input
//!   scalers and input sets, so inputs are concatenated rather than shared), and the head has one
//!   output per member.
//! * [`EnsembleLayout::Batched`]: one lane per member, zero-padded to the largest input/hidden
//!   size (the `vmap` / `stack_module_state` approach).
//! * [`EnsembleLayout::BatchedByHidden`]: like `Batched`, but one stack per distinct hidden size
//!   so no hidden units are padded.
//! * [`EnsembleLayout::PerModel`]: one stack per member; equivalent to running members separately.
//!
//! Zero padding is exact. A padded input column has zero weights. A padded hidden unit has zero
//! weights and biases, so every step its gates are `i = f = o = sigmoid(0)`, `g = tanh(0) = 0`;
//! starting from `c = 0` it stays at `c = 0, h = 0`, and its head weight is zero.
//!
//! The head emits one value per member rather than the ensemble mean: the BMI denormalises each
//! member with its own scaler and clamps at zero before averaging, and that clamp is nonlinear.

use std::collections::BTreeMap;
use std::str::FromStr;

use burn::prelude::*;
use burn::tensor::activation::{sigmoid, tanh};

use crate::nextgen_lstm::NextgenLstm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnsembleLayout {
    PerModel,
    BlockDiagonal,
    Batched,
    BatchedByHidden,
}

impl EnsembleLayout {
    pub const ALL: [EnsembleLayout; 4] = [
        EnsembleLayout::PerModel,
        EnsembleLayout::BlockDiagonal,
        EnsembleLayout::Batched,
        EnsembleLayout::BatchedByHidden,
    ];

    pub fn name(self) -> &'static str {
        match self {
            EnsembleLayout::PerModel => "per-model",
            EnsembleLayout::BlockDiagonal => "block-diagonal",
            EnsembleLayout::Batched => "batched",
            EnsembleLayout::BatchedByHidden => "batched-by-hidden",
        }
    }
}

impl FromStr for EnsembleLayout {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|layout| layout.name() == s)
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|l| l.name()).collect();
                format!("unknown ensemble layout {s:?}, expected one of {names:?}")
            })
    }
}

/// Plain f32 copy of one member's weights, in burn's `[d_input, d_output]` layout.
struct MemberWeights {
    input_size: usize,
    hidden_size: usize,
    w_ih: Vec<f32>,   // [input, 4 * hidden], gate blocks i, f, g, o along the columns
    b_ih: Vec<f32>,   // [4 * hidden]
    w_hh: Vec<f32>,   // [hidden, 4 * hidden]
    b_hh: Vec<f32>,   // [4 * hidden]
    head_w: Vec<f32>, // [hidden] (single output)
    head_b: f32,
}

fn to_vec<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .try_to_vec::<f32>()
        .expect("model parameters are f32")
}

impl MemberWeights {
    fn from_model(model: &NextgenLstm) -> Self {
        let lstm = &model.lstm;
        let [input_size, gate_width] = lstm.input_transform.weight.dims();
        let hidden_size = lstm.d_hidden;
        assert_eq!(gate_width, 4 * hidden_size);
        let [_, outputs] = model.head.weight.dims();
        assert_eq!(outputs, 1, "stacked ensembles expect single-output members");
        let bias = |linear: &burn::nn::Linear| {
            to_vec(linear.bias.as_ref().expect("LSTM layers have biases").val())
        };
        Self {
            input_size,
            hidden_size,
            w_ih: to_vec(lstm.input_transform.weight.val()),
            b_ih: bias(&lstm.input_transform),
            w_hh: to_vec(lstm.hidden_transform.weight.val()),
            b_hh: bias(&lstm.hidden_transform),
            head_w: to_vec(model.head.weight.val()),
            head_b: bias(&model.head)[0],
        }
    }
}

/// Where one ensemble member lives inside a [`Stack`].
#[derive(Clone, Copy, Debug)]
struct Slot {
    member: usize,
    lane: usize,
    input_offset: usize,
    hidden_offset: usize,
    output: usize,
}

/// `lanes` independent LSTMs of identical (padded) shape, run with batched matmuls.
#[derive(Debug)]
struct Stack {
    lanes: usize,
    input_width: usize,
    hidden_width: usize,
    outputs: usize,
    w_ih: Tensor<3>,   // [lanes, input_width, 4 * hidden_width]
    b_ih: Tensor<3>,   // [lanes, 1, 4 * hidden_width]
    w_hh: Tensor<3>,   // [lanes, hidden_width, 4 * hidden_width]
    b_hh: Tensor<3>,   // [lanes, 1, 4 * hidden_width]
    head_w: Tensor<3>, // [lanes, hidden_width, outputs]
    head_b: Tensor<3>, // [lanes, 1, outputs]
    slots: Vec<Slot>,
}

impl Stack {
    fn build(members: &[MemberWeights], slots: Vec<Slot>, device: &Device) -> Self {
        let lanes = slots.iter().map(|s| s.lane + 1).max().unwrap_or(0);
        let outputs = slots.iter().map(|s| s.output + 1).max().unwrap_or(0);
        let input_width = slots
            .iter()
            .map(|s| s.input_offset + members[s.member].input_size)
            .max()
            .unwrap_or(0);
        let hidden_width = slots
            .iter()
            .map(|s| s.hidden_offset + members[s.member].hidden_size)
            .max()
            .unwrap_or(0);
        let gates = 4 * hidden_width;

        let mut w_ih = vec![0f32; lanes * input_width * gates];
        let mut b_ih = vec![0f32; lanes * gates];
        let mut w_hh = vec![0f32; lanes * hidden_width * gates];
        let mut b_hh = vec![0f32; lanes * gates];
        let mut head_w = vec![0f32; lanes * hidden_width * outputs];
        let mut head_b = vec![0f32; lanes * outputs];

        for slot in &slots {
            let m = &members[slot.member];
            let h = m.hidden_size;
            // Column of unit `j` of gate `q` of this member in the stacked gate dimension.
            let col = |q: usize, j: usize| q * hidden_width + slot.hidden_offset + j;
            for q in 0..4 {
                for j in 0..h {
                    let src = q * h + j;
                    let dst = col(q, j);
                    b_ih[slot.lane * gates + dst] = m.b_ih[src];
                    b_hh[slot.lane * gates + dst] = m.b_hh[src];
                    for r in 0..m.input_size {
                        let row = slot.input_offset + r;
                        w_ih[(slot.lane * input_width + row) * gates + dst] =
                            m.w_ih[r * 4 * h + src];
                    }
                    for r in 0..h {
                        let row = slot.hidden_offset + r;
                        w_hh[(slot.lane * hidden_width + row) * gates + dst] =
                            m.w_hh[r * 4 * h + src];
                    }
                }
            }
            for (r, &weight) in m.head_w.iter().enumerate() {
                let row = slot.hidden_offset + r;
                head_w[(slot.lane * hidden_width + row) * outputs + slot.output] = weight;
            }
            head_b[slot.lane * outputs + slot.output] = m.head_b;
        }

        let tensor = |data: Vec<f32>, shape: [usize; 3]| {
            Tensor::<3>::from_data(TensorData::new(data, shape), device)
        };
        Self {
            lanes,
            input_width,
            hidden_width,
            outputs,
            w_ih: tensor(w_ih, [lanes, input_width, gates]),
            b_ih: tensor(b_ih, [lanes, 1, gates]),
            w_hh: tensor(w_hh, [lanes, hidden_width, gates]),
            b_hh: tensor(b_hh, [lanes, 1, gates]),
            head_w: tensor(head_w, [lanes, hidden_width, outputs]),
            head_b: tensor(head_b, [lanes, 1, outputs]),
            slots,
        }
    }

    /// One timestep. `input` is `[lanes, 1, input_width]`; returns `[lanes, 1, outputs]`.
    fn step(&self, input: Tensor<3>, state: &mut Option<(Tensor<3>, Tensor<3>)>) -> Tensor<3> {
        let (cell, hidden) = state.take().unwrap_or_else(|| {
            let zeros = Tensor::zeros([self.lanes, 1, self.hidden_width], &input.device());
            (zeros.clone(), zeros)
        });

        let gates = (input.matmul(self.w_ih.clone()) + self.b_ih.clone())
            + (hidden.matmul(self.w_hh.clone()) + self.b_hh.clone());
        let [i, f, g, o]: [Tensor<3>; 4] = gates
            .chunk(4, 2)
            .try_into()
            .expect("gate dimension is 4 * hidden_width");

        let cell = sigmoid(f) * cell + sigmoid(i) * tanh(g);
        let hidden = sigmoid(o) * tanh(cell.clone());
        let output = hidden.clone().matmul(self.head_w.clone()) + self.head_b.clone();
        *state = Some((cell, hidden));
        output
    }
}

/// An ensemble of [`NextgenLstm`] members run as stacked LSTMs. Carries its own recurrent state.
#[derive(Debug)]
pub struct StackedEnsemble {
    layout: EnsembleLayout,
    members: usize,
    stacks: Vec<Stack>,
    states: Vec<Option<(Tensor<3>, Tensor<3>)>>,
}

impl StackedEnsemble {
    pub fn new(models: &[&NextgenLstm], layout: EnsembleLayout, device: &Device) -> Self {
        let members: Vec<MemberWeights> = models
            .iter()
            .map(|m| MemberWeights::from_model(m))
            .collect();
        let slot = |member, lane, input_offset, hidden_offset, output| Slot {
            member,
            lane,
            input_offset,
            hidden_offset,
            output,
        };

        let groups: Vec<Vec<Slot>> = match layout {
            EnsembleLayout::PerModel => (0..members.len())
                .map(|k| vec![slot(k, 0, 0, 0, 0)])
                .collect(),
            EnsembleLayout::Batched => {
                vec![(0..members.len()).map(|k| slot(k, k, 0, 0, 0)).collect()]
            }
            EnsembleLayout::BatchedByHidden => {
                let mut by_hidden: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for (k, m) in members.iter().enumerate() {
                    by_hidden.entry(m.hidden_size).or_default().push(k);
                }
                by_hidden
                    .into_values()
                    .map(|ks| {
                        ks.into_iter()
                            .enumerate()
                            .map(|(lane, k)| slot(k, lane, 0, 0, 0))
                            .collect()
                    })
                    .collect()
            }
            EnsembleLayout::BlockDiagonal => {
                let (mut input_offset, mut hidden_offset) = (0, 0);
                vec![
                    members
                        .iter()
                        .enumerate()
                        .map(|(k, m)| {
                            let s = slot(k, 0, input_offset, hidden_offset, k);
                            input_offset += m.input_size;
                            hidden_offset += m.hidden_size;
                            s
                        })
                        .collect(),
                ]
            }
        };

        let stacks: Vec<Stack> = groups
            .into_iter()
            .map(|slots| Stack::build(&members, slots, device))
            .collect();
        Self {
            layout,
            members: members.len(),
            states: (0..stacks.len()).map(|_| None).collect(),
            stacks,
        }
    }

    pub fn layout(&self) -> EnsembleLayout {
        self.layout
    }

    /// Clears the recurrent state back to zeros.
    pub fn reset(&mut self) {
        self.states.iter_mut().for_each(|s| *s = None);
    }

    /// Advances every member one timestep. `inputs[k]` is member `k`'s (scaled) input vector;
    /// returns each member's raw (normalised) output, in member order.
    pub fn step(&mut self, inputs: &[Vec<f32>], device: &Device) -> Vec<f32> {
        assert_eq!(inputs.len(), self.members);
        let mut outputs = vec![0f32; self.members];
        for (stack, state) in self.stacks.iter().zip(&mut self.states) {
            let mut x = vec![0f32; stack.lanes * stack.input_width];
            for slot in &stack.slots {
                let start = slot.lane * stack.input_width + slot.input_offset;
                let member_input = &inputs[slot.member];
                x[start..start + member_input.len()].copy_from_slice(member_input);
            }
            let x = Tensor::<3>::from_data(
                TensorData::new(x, [stack.lanes, 1, stack.input_width]),
                device,
            );
            let y = to_vec(stack.step(x, state));
            for slot in &stack.slots {
                outputs[slot.member] = y[slot.lane * stack.outputs + slot.output];
            }
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::LstmState;

    /// Every layout must reproduce running the members one by one.
    #[test]
    fn layouts_match_separate_members() {
        let device = Device::flex();
        // (inputs, hidden): mixed input sets and hidden sizes, like the real ensembles.
        let shapes = [(4, 3), (5, 3), (2, 6), (5, 6)];
        let models: Vec<NextgenLstm> = shapes
            .iter()
            .map(|&(i, h)| NextgenLstm::init(&device, i, h, 1))
            .collect();
        let refs: Vec<&NextgenLstm> = models.iter().collect();
        let inputs = |t: usize| -> Vec<Vec<f32>> {
            shapes
                .iter()
                .enumerate()
                .map(|(k, &(i, _))| {
                    (0..i)
                        .map(|j| ((t * 7 + k * 3 + j) as f32 * 0.37).sin())
                        .collect()
                })
                .collect()
        };

        let mut states: Vec<Option<LstmState<2>>> = vec![None; models.len()];
        let mut expected = Vec::new();
        for t in 0..30 {
            let step: Vec<f32> = models
                .iter()
                .zip(&mut states)
                .zip(inputs(t))
                .map(|((model, state), x)| {
                    let n = x.len();
                    let x = Tensor::<3>::from_data(TensorData::new(x, [1, 1, n]), &device);
                    let (y, s) = model.forward(x, state.take());
                    *state = Some(s);
                    to_vec(y)[0]
                })
                .collect();
            expected.push(step);
        }

        for layout in EnsembleLayout::ALL {
            let mut ensemble = StackedEnsemble::new(&refs, layout, &device);
            for (t, want) in expected.iter().enumerate() {
                let got = ensemble.step(&inputs(t), &device);
                for (g, w) in got.iter().zip(want) {
                    assert!(
                        (g - w).abs() <= 1e-6,
                        "{layout:?} step {t}: {got:?} != {want:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn layout_names_round_trip() {
        for layout in EnsembleLayout::ALL {
            assert_eq!(layout.name().parse::<EnsembleLayout>(), Ok(layout));
        }
        assert!("nope".parse::<EnsembleLayout>().is_err());
    }
}
