// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Demonstrates guarded Sentry initialisation behind the `sentry` feature flag.
//
// Run with:
//   SENTRY_DSN=https://examplePublicKey@o0.ingest.sentry.io/0 cargo run --features sentry --example sentry_init

use hybrid_fusion::{
    HybridConfig, HybridNetwork, NeuroModulators, Result, SpikingNetwork, Tensor, Transformer,
};

struct DemoTransformer {
    dim: usize,
    max_seq: usize,
}

impl Transformer for DemoTransformer {
    fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        let seq = token_ids.len();
        let data: Vec<f32> = (0..seq * self.dim).map(|i| (i as f32) * 0.001).collect();
        Tensor::from_vec(data, &[seq, self.dim])
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq
    }
    fn param_count(&self) -> usize {
        self.dim * 12_000
    }
}

struct DemoSnn {
    channels: usize,
}

impl SpikingNetwork for DemoSnn {
    fn step(&mut self, stimuli: &[f32], _modulators: &NeuroModulators) -> Result<Vec<usize>> {
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

fn main() -> Result<()> {
    // Guarded Sentry init: only activates if SENTRY_DSN is set.
    let _guard = sentry::init((
        std::env::var("SENTRY_DSN").unwrap_or_default(),
        sentry::ClientOptions {
            release: sentry::release_name!(),
            ..Default::default()
        },
    ));

    let cfg = HybridConfig::tiny();
    let transformer = DemoTransformer {
        dim: cfg.transformer.dim,
        max_seq: cfg.transformer.max_seq_len,
    };
    let snn = DemoSnn {
        channels: cfg.snn_input_channels,
    };
    let mut net = HybridNetwork::new(transformer, snn, cfg);

    let token_ids: Vec<u32> = vec![1, 2, 3, 4];
    let out = net.forward(&token_ids, None)?;
    println!(
        "step {} | fired {} neurons | stimuli[0] = {:.4}",
        out.global_step,
        out.fired_neurons.len(),
        out.stimuli[0],
    );

    // Example: capture an error event (only sent when SENTRY_DSN is set).
    sentry::capture_message("hybrid-fusion demo completed", sentry::Level::Info);

    Ok(())
}
