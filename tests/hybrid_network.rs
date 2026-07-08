// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for `HybridNetwork` using mock backends.
//!
//! These tests exercise only the public API surface — no `pub(crate)` or internal access —
//! and serve as usage examples for consumers of the crate.

use hybrid_fusion::{
    HybridConfig, HybridError, HybridNetwork, HybridOutput, NeuroModulators, SpikingNetwork,
    Tensor, Transformer,
};

// ---------------------------------------------------------------------------
// Mock backends
// ---------------------------------------------------------------------------

struct MockTransformer {
    dim: usize,
    max_seq: usize,
}

impl Transformer for MockTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq = token_ids.len();
        let data: Vec<f32> = (0..seq * self.dim).map(|i| (i as f32) * 0.01).collect();
        Tensor::from_vec(data, &[seq, self.dim])
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq
    }

    fn param_count(&self) -> usize {
        self.dim * 1000
    }
}

struct MockSnn {
    channels: usize,
}

impl SpikingNetwork for MockSnn {
    fn step(&mut self, stimuli: &[f32], _modulators: &NeuroModulators) -> hybrid_fusion::Result<Vec<usize>> {
        Ok(stimuli
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(i, _)| i)
            .collect())
    }

    fn num_channels(&self) -> usize {
        self.channels
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_network() -> HybridNetwork<MockTransformer, MockSnn> {
    let cfg = HybridConfig::tiny();
    let t = MockTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let s = MockSnn {
        channels: cfg.snn_input_channels,
    };
    HybridNetwork::new(t, s, cfg)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_forward_output_shapes() {
    let cfg = HybridConfig::tiny();
    let mut net = build_network();

    let out: HybridOutput = net.forward(&[1, 2, 3, 4], None).expect("forward should succeed");

    // embedding dimension matches the transformer's hidden dim
    assert_eq!(out.embedding.len(), cfg.transformer.dim);

    // stimuli dimension matches the SNN's channel count
    assert_eq!(out.stimuli.len(), cfg.snn_input_channels);

    // fired_neurons is a subset of valid channel indices
    for &idx in &out.fired_neurons {
        assert!(idx < cfg.snn_input_channels, "fired neuron index out of range");
    }

    // first forward sets global_step to 1
    assert_eq!(out.global_step, 1);
}

#[test]
fn test_forward_rejects_empty() {
    let mut net = build_network();

    let err = net.forward(&[], None).expect_err("empty token_ids should fail");
    match err {
        HybridError::InputLengthMismatch { expected, got } => {
            assert_eq!(expected, 1);
            assert_eq!(got, 0);
        }
        other => panic!("expected InputLengthMismatch, got {other:?}"),
    }
}

#[test]
fn test_forward_rejects_over_long() {
    let cfg = HybridConfig::tiny();
    let mut net = build_network();
    let too_long = vec![0u32; cfg.transformer.max_seq_len + 1];

    let err = net
        .forward(&too_long, None)
        .expect_err("over-length token_ids should fail");
    match err {
        HybridError::InputLengthMismatch { expected, got } => {
            assert_eq!(expected, cfg.transformer.max_seq_len);
            assert_eq!(got, too_long.len());
        }
        other => panic!("expected InputLengthMismatch, got {other:?}"),
    }
}

#[test]
fn test_global_step_increments_and_resets() {
    let mut net = build_network();

    assert_eq!(net.global_step(), 0, "initial step should be 0");

    net.forward(&[0, 1], None).unwrap();
    assert_eq!(net.global_step(), 1);

    net.forward(&[0, 1], None).unwrap();
    assert_eq!(net.global_step(), 2, "step should increment each forward");

    net.reset();
    assert_eq!(net.global_step(), 0, "reset should bring step back to 0");
}

#[test]
fn test_stimuli_bounded() {
    let mut net = build_network();

    // Use a variety of token counts to exercise different pooling paths
    for tokens in [1u32, 3, 8, 32] {
        let token_ids: Vec<u32> = (0..tokens).collect();
        let out = net.forward(&token_ids, None).unwrap();

        for (i, v) in out.stimuli.iter().enumerate() {
            assert!(
                v.abs() <= 1.0,
                "stimuli[{i}] = {v} is outside [-1, 1] for {tokens} tokens"
            );
        }
    }
}

#[test]
fn test_forward_with_custom_modulators() {
    let mut net = build_network();

    let custom = NeuroModulators {
        dopamine: 0.8,
        cortisol: 0.2,
        acetylcholine: 0.6,
        tempo: 1.5,
        aux_dopamine: 0.1,
    };

    let out = net
        .forward(&[10, 20, 30], Some(custom))
        .expect("forward with custom modulators should succeed");

    // basic shape checks still hold
    assert_eq!(out.embedding.len(), net.transformer.dim());
    assert_eq!(out.stimuli.len(), net.snn.num_channels());
    assert_eq!(out.global_step, 1);

    // stimuli must still be bounded even with non-default modulators
    for v in &out.stimuli {
        assert!(v.abs() <= 1.0);
    }
}
