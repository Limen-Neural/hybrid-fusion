# REVIEW.md

Code review guidelines for hybrid-fusion.

## Architecture reference

The core data flow is:

```text
token_ids -> Transformer::hidden_states -> projector -> tanh -> SpikingNetwork::step -> fired_neurons
```

`hybrid-fusion` is a **pure orchestration crate** — it defines traits and wires them together. It must never depend on concrete SNN or Transformer implementations.

## Review checklist

Copy-paste this into every PR review:

### Trait boundaries

- [ ] No concrete `cortex-tensor`, `neuromod`, or backend-specific types in public API
- [ ] `Transformer` and `SpikingNetwork` traits remain the only coupling points
- [ ] No new `use` statements pulling in concrete backend crates
- [ ] Public types are generic over trait bounds, not pinned to implementations
- [ ] Reference: LIM-9 boundary matrix for ownership rules

### Invariants

- [ ] Stimuli values are always in `[-1, 1]` (enforced by `tanh` in projector)
- [ ] Tensor dimensions are always > 0 (reject zero-dim tensors)
- [ ] `token_ids` is never empty (validated in `HybridNetwork::forward`)
- [ ] `token_ids.len()` never exceeds `transformer.max_seq_len()`
- [ ] No silent fallback on shape mismatches — return an error

### Safety

- [ ] No `unsafe` code unless explicitly justified with a safety comment
- [ ] No new dependencies on concrete SNN or Transformer implementations
- [ ] Feature gates used correctly for optional functionality

### Testing

- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy -- -D warnings` is clean
- [ ] New code has corresponding tests
- [ ] Tests cover happy path and edge cases (empty input, max-length, zero-dim rejection)

### Documentation

- [ ] SPDX license header on all `.rs` files
- [ ] Doc comments on all public items (`///` for items, `//!` for modules)
- [ ] Breaking changes documented in CHANGELOG.md

## What to look for in trait changes

Changes to `Transformer`, `SpikingNetwork`, or `NeuroModulators` require extra scrutiny:

1. **Backward compatibility** — adding a method with a default impl is safe; removing or renaming one breaks every downstream implementor
2. **Semantic contracts** — does the new method carry invariants (e.g., output bounds, shape requirements)? Document them in the trait-level doc comment
3. **LIM-9 ownership** — verify the owning team is listed in the boundary matrix before merging trait changes
4. **Downstream impact** — check if `neuromod`, `cortex-tensor`, or any SNN backend implements the trait and would break

## Boundary matrix

The LIM-9 boundary matrix is the source of truth for which team owns which trait and which crate may depend on which. Consult it before merging any change that touches:

- Trait definitions in `src/traits.rs`
- Public re-exports in `src/lib.rs`
- `Cargo.toml` dependencies

## Common review comments

### Edition 2024 is intentional

Do not suggest downgrading to edition 2021. This project uses edition 2024.

### tanh is the boundary enforcer

The projector applies `tanh` to guarantee stimuli are in `[-1, 1]`. If you see code that bypasses the projector to feed raw values into `SpikingNetwork::step`, flag it.

### Zero-dim tensors are a bug

If a code path produces a zero-dimensional tensor, that is always an error — never silently propagate it.

## AI reviewer notes

### Codacy

- Flags markdown lint issues — check CHANGELOG formatting
- May flag `println!` — prefer `eprintln!` or logging in library code

### Gemini

- May flag relative links in doc comments — use plain text license references
- Generally accurate on code quality suggestions

### CodeRabbit

- Auto-resolves threads when fix is pushed — verify thread status before merging
- Sometimes flags pre-existing issues — confirm the issue was introduced by the PR

## CI/CD reference

Run locally before pushing:

```bash
cargo check                 # compile check
cargo test --all-features   # all unit + doc tests
cargo clippy -- -D warnings # lint, deny all warnings
```
