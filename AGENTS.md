# AGENTS.md — hybrid-fusion

Context file for AI coding agents working in this repository.

## Crate Purpose

`hybrid-fusion` is a **backend-agnostic pure-Rust orchestrator** for hybrid Transformer <-> Spiking Neural Network (SNN) forward-pass pipelines. It depends only on trait abstractions — no concrete math, neuron dynamics, or model-loading implementations live here.

## Crate Boundaries

### This crate OWNS

- Orchestration of hybrid ANN -> SNN forward-pass paths
- Transformer hidden-state pooling / resizing into bounded SNN stimuli
- The public `HybridNetwork<T, S>` API and its error boundaries
- Projector logic (dimensionality reduction from transformer hidden space to SNN input channels)
- Configuration and output types (`HybridConfig`, `TransformerConfig`, `HybridOutput`)

### This crate does NOT own

| Concern | Owned by |
|---|---|
| Tensor / transformer math | `cortex-tensor` |
| GGUF model parsing | `engram-parser` |
| SNN neuron dynamics (LIF, Izhikevich) | `neuromod` |
| SNN runtime / scheduling | `brainstem-daemon` |
| Neuromodulator mapping / critic signals | `limbic-critic` |
| Sensory encoding | `axon-encoder` |

**Rule:** This crate MUST NOT depend on concrete implementations. All backend interaction flows through the trait contracts defined in `src/traits.rs`.

## Trait Contracts

Three core traits define the pluggable backend surface:

- **`Transformer`** — produces hidden-state tensors from token IDs. Implementations live in downstream crates (e.g. `cortex-tensor`).
- **`SpikingNetwork`** — steps the SNN forward given stimuli + neuromodulator state, returns fired neuron indices. Implementations live in `neuromod` / `brainstem-daemon`.
- **`GgufLoader`** — loads GGUF model layouts from disk. Implementations live in `engram-parser`.

`HybridNetwork<T: Transformer, S: SpikingNetwork>` is generic over the `Transformer` and `SpikingNetwork` traits; `GgufLoader` is consumed separately. Any code that adds a concrete backend dependency to `Cargo.toml` is a boundary violation.

## Sibling Crates

| Crate | Responsibility |
|---|---|
| `neuromod` | SNN neuron dynamics (LIF, Izhikevich models) |
| `cortex-tensor` | Tensor math and transformer inference |
| `engram-parser` | GGUF file parsing and model layout extraction |
| `brainstem-daemon` | SNN runtime scheduling and execution |
| `limbic-critic` | Neuromodulator-to-critic signal mapping |
| `axon-encoder` | Sensory input encoding |

## Build & Test

```bash
cargo build
cargo test --all-features
cargo clippy -- -D warnings
```

## Org Boundary Matrix

See [LIM-9](https://github.com/Limen-Neural/hybrid-fusion/issues/5) for the full boundary matrix (cross-crate ownership and dependency rules across the Limen-Neural organisation).
