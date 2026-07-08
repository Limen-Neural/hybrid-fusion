# hybrid-fusion

[![CI](https://github.com/Limen-Neural/hybrid-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/hybrid-fusion/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**Pure-Rust master orchestrator for a hybrid transformer <-> spiking neural
network stack.** Zero Candle, zero CUDA, zero Julia.

`hybrid-fusion` defines the orchestration contract for piping transformer
hidden states into spiking neural network dynamics. It is intentionally
**backend-agnostic**: transformer and SNN implementations are injected via
traits (`Transformer`, `SpikingNetwork`), so downstream consumers choose
their own math engines.

## Architecture

```text
token_ids: &[u32]
     |
     v  Transformer::hidden_states
Tensor   [seq_len, dim]
     |
     v  projector::embed_to_stimuli_with_width   (pool -> resize -> tanh)
stimuli: Vec<f32>       in [-1, 1], length == snn.num_channels()
     |
     v  SpikingNetwork::step(&stimuli, &modulators)
fired_neurons: Vec<usize>
```

The `tanh` squash is applied **after** pooling and resizing so the values
fed into the SNN are always bounded.

## Scope / Boundaries

This crate **owns**:

- Orchestration of hybrid ANN -> SNN forward-pass paths.
- Transformer hidden-state pooling and resizing into bounded SNN stimuli.
- The public `HybridNetwork<T, S>` API and error boundaries.

This crate **does not own**:

- Tensor/transformer/MoE math -> see [`cortex-tensor`](https://github.com/Spikenaut/cortex-tensor).
- GGUF parsing and weight layout -> see [`engram-parser`](https://github.com/Spikenaut/engram-parser).
- Neuron dynamics and SNN integration internals -> see [`neuromod`](https://github.com/Spikenaut/neuromod).

See [LIM-9](https://github.com/Limen-Neural/hybrid-fusion/issues/5) for the
full boundary matrix.

## Quick start

```rust
use hybrid_fusion::{HybridConfig, HybridNetwork, NeuroModulators, Transformer, SpikingNetwork};
use hybrid_fusion::Tensor;
use hybrid_fusion::Result;

// Implement traits for your backend, then wire them:
// let mut net = HybridNetwork::new(my_transformer, my_snn, HybridConfig::tiny());
// let out = net.forward(&[1u32, 2, 3, 4], None)?;
```

## Public surface

| Item | Purpose |
|------|---------|
| `HybridNetwork<T, S>` | Generic orchestrator over any `Transformer` + `SpikingNetwork`. |
| `Transformer` trait | Backend-agnostic transformer interface. |
| `SpikingNetwork` trait | Backend-agnostic SNN interface. |
| `NeuroModulators` | Neuromodulator struct passed to SNN steps. |
| `HybridConfig` / `TransformerConfig` | Predefined configs (`tiny`, `olmo_1b`). |
| `projector::embed_to_stimuli_with_width` | Pool -> resize -> tanh adapter. |
| `Tensor` | Lightweight owned tensor (data + shape). |

## Guides

- **[Implementing a Backend](docs/implementing-backends.md)** — trait contracts, data flow, tensor shape conventions, and a minimal working example for `Transformer` + `SpikingNetwork`.
## Error monitoring (optional)

`hybrid-fusion` ships an optional
[Sentry](https://docs.sentry.io/platforms/rust/) integration for error
reporting in downstream services.

### Enabling

Add the feature flag when building:

```sh
cargo build --features sentry
```

### Configuration

Set the `SENTRY_DSN` environment variable to your Sentry project DSN.
**Do not commit DSNs** — pass them at runtime or via a secrets manager.

```sh
export SENTRY_DSN=https://examplePublicKey@o0.ingest.sentry.io/0
```

### Initialisation pattern

Guard the init call so the application starts cleanly even without a DSN:

```rust
let _guard = sentry::init((
    std::env::var("SENTRY_DSN").unwrap_or_default(),
    sentry::ClientOptions {
        release: sentry::release_name!(),
        ..Default::default()
    },
));
```

The `_guard` must be held for the lifetime of the application — dropping it
flushes pending events and shuts down the transport.

See [`examples/sentry_init.rs`](examples/sentry_init.rs) for a full working
example.

## Status

Experimental. API is expected to change as backend crates evolve.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)
at your option.
