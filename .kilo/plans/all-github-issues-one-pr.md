# Plan: Resolve All Open GitHub Issues In One PR

## Scope

Repository: `Limen-Neural/hybrid-fusion` at `/home/raulmc/Limen-Neural/hybrid-fusion/hybrid-fusion`.

Open GitHub issues to include in one PR:

1. `#3` Add GitHub Actions CI workflow for code validation.
2. `#4` Switch from `GPL-3.0-or-later` to dual `MIT OR Apache-2.0`.
3. `#5` Document `hybrid-fusion` role in the LIM-9 boundary matrix.

The PR should use one feature branch from `main`, with a PR body containing `Closes #3`, `Closes #4`, and `Closes #5`.

## Current State Observations

The repo already has `.github/workflows/rust.yml`, but it only runs `cargo build --verbose` and `cargo test --verbose`. Issue `#3` asks for `.github/workflows/ci.yml` with rustfmt, clippy, build, and test.

`Cargo.toml` currently declares `license = "GPL-3.0-or-later"` and uses local path dependencies for `cortex-tensor`, `engram-parser`, and `neuromod`. The checked-out project is nested, so the current relative paths do not appear valid from the repo root. They also would not work in GitHub Actions unless those sibling repos are available in the runner.

`LICENSE` currently contains the GPL-3.0 text. There are no `LICENSE-MIT`, `LICENSE-APACHE`, or `CHANGELOG` files.

Rust source files do not currently have SPDX headers. The issue asks for SPDX in `.rs` files.

`README.md` already describes the orchestrator and cross-references the three sibling crates, but it does not yet have a dedicated Scope/Boundaries section, a LIM-9 link, dual-license badge/text, or changelog/publish notes.

There is no beads database in the nested repo, so GitHub issues are the active tracker for this work.

## Recommended Approach

Use a single consolidated maintenance branch, for example `chore/resolve-open-issues`, and keep changes grouped by issue area:

1. CI and dependency resolution.
2. Licensing and release metadata.
3. README boundary documentation.

This is better than separate branches because the user explicitly requested one PR and the changes are compatible: CI should validate the licensing/docs updates, and all three issues are small enough for one review.

## Implementation Steps

1. Prepare branch and baseline

Create a branch from latest `origin/main`, after verifying the nested repo working tree. Do not touch unrelated files outside `/home/raulmc/Limen-Neural/hybrid-fusion/hybrid-fusion`.

2. Resolve dependency setup for CI

Update `Cargo.toml` so dependencies are usable by GitHub Actions. Prefer Git dependencies pointing at the sibling repositories, with versions retained where appropriate, because GitHub-hosted CI cannot see local sibling paths. If local path development is still needed, use an explicit, non-committed local override strategy rather than committing broken relative paths.

Expected dependency direction:

```toml
cortex-tensor = { git = "https://github.com/Limen-Neural/cortex-tensor" }
engram-parser = { git = "https://github.com/Limen-Neural/engram-parser" }
neuromod = { version = "0.5.0", git = "https://github.com/Limen-Neural/neuromod" }
```

If `neuromod` API compatibility fails with `0.5.0`, either pin the Git revision that matches the existing API or use the latest compatible public release. The key acceptance target is that clean checkout CI can build without unavailable local paths.

3. Add requested CI workflow

Create `.github/workflows/ci.yml` matching issue `#3` intent:

- Trigger on pushes and PRs to `main`.
- Install Rust stable with `rustfmt` and `clippy`.
- Cache Cargo registry, Git dependencies, and `target`.
- Run `cargo fmt --check`.
- Run `cargo clippy --all-targets --all-features -- -D warnings`.
- Run `cargo build --all-features`.
- Run `cargo test --all-features`.

Handle existing `.github/workflows/rust.yml` by either removing it or reducing duplication. Preferred: replace it with `ci.yml` and delete `rust.yml`, because leaving both workflows would duplicate build/test checks and could confuse reviewers.

4. Fix formatting and clippy warnings

Run the same commands locally after implementation. Address only warnings needed for `-D warnings`; avoid unrelated refactors.

Likely areas to inspect if clippy fails:

- Long chained lines in `src/hybrid.rs`.
- Unused variants or imports after dependency/API adjustments.
- Example code style in `examples/hybrid_telemetry.rs`.

5. Migrate license metadata

Update package license in `Cargo.toml` to:

```toml
license = "MIT OR Apache-2.0"
```

Replace the current GPL-only license setup with dual license files:

- `LICENSE-MIT`
- `LICENSE-APACHE`

Consider whether to keep `LICENSE` as a short pointer to both files or remove/rename it. Preferred for Rust crates: keep a concise `LICENSE` notice that points to `LICENSE-MIT` and `LICENSE-APACHE`, while the authoritative full texts live in the two dedicated files. This makes repository landing-page license behavior clearer without leaving GPL text behind.

6. Add SPDX headers

Add SPDX headers to all Rust source files in scope:

- `src/lib.rs`
- `src/error.rs`
- `src/hybrid.rs`
- `src/projector.rs`
- `src/types.rs`
- `examples/hybrid_telemetry.rs`

Use the minimal header:

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0
```

Place it before module-level docs so the first line clearly states license metadata.

7. Update README for licensing and boundaries

Update `README.md` with:

- License badge or clear dual-license line near the top.
- Scope/Boundaries section satisfying issue `#5`.
- Explicit LIM-9 link. If a canonical URL is not present in the issue, use a GitHub issue/reference link format such as `LIM-9` with the best available repository reference, and keep the wording factual.
- Cross-references to `cortex-tensor`, `engram-parser`, and `neuromod` as boundary neighbors.
- License section changed from GPL-only to `MIT OR Apache-2.0`.

Recommended Scope/Boundaries content:

- Owns orchestration of hybrid ANN to SNN flow.
- Owns transformer hidden-state pooling/resizing into bounded SNN stimuli.
- Owns public `HybridNetwork` API and error boundaries across the three crates.
- Does not own tensor/transformer/MoE math; that belongs to `cortex-tensor`.
- Does not own GGUF parsing/weight layout; that belongs to `engram-parser`.
- Does not own neuron dynamics or SNN integration internals; that belongs to `neuromod`.

8. Add changelog entry

Create `CHANGELOG.md` with an initial entry for the upcoming release or `Unreleased` section. Include:

- License migration to dual MIT/Apache-2.0.
- CI validation workflow.
- Boundary-matrix documentation.

Keep the changelog concise and conventional; do not invent historical release notes.

9. Add publish-step documentation

Satisfy the issue `#4` publish-step checklist with a short README or changelog note describing the pre-publish validation sequence:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo publish --dry-run
```

If dependency Git sources make `cargo publish --dry-run` impossible, document the blocker and either avoid claiming publish readiness or adjust dependencies to crates.io-compatible versions before finalizing.

10. Validate locally

Run quality gates from the nested repo root:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-features
cargo test --all-features
```

If the initial dependency changes require generating or updating `Cargo.lock`, commit it only if the crate policy favors committed lockfiles. For a library crate, omitting `Cargo.lock` is common, but CI cache key can still safely use `hashFiles('**/Cargo.lock')` even when absent.

11. Commit and create PR

Commit the changes with a conventional message such as:

```text
chore: resolve open project setup issues
```

Push the branch and create one PR against `main` using `gh pr create`. PR title recommendation:

```text
chore: add CI, dual licensing, and boundary docs
```

PR body should summarize:

- Added full Rust CI workflow.
- Migrated crate licensing to `MIT OR Apache-2.0` with SPDX headers.
- Documented LIM-9 boundary role and sibling crate ownership.

Include:

```text
Closes #3
Closes #4
Closes #5
```

## Validation Criteria

The work is ready when:

1. `cargo fmt --check` passes.
2. `cargo clippy --all-targets --all-features -- -D warnings` passes.
3. `cargo build --all-features` passes.
4. `cargo test --all-features` passes.
5. `.github/workflows/ci.yml` exists and is the active validation workflow.
6. License files and `Cargo.toml` consistently say `MIT OR Apache-2.0`.
7. Every Rust source/example file has an SPDX header.
8. README includes Scope/Boundaries, LIM-9 reference, sibling crate cross-references, and dual-license text.
9. Changelog records the license/CI/docs changes.
10. The PR is open and references all three issues with closing keywords.

## Risks And Decisions To Watch

Dependency source strategy is the main risk. Git dependencies are best for GitHub Actions in the current repo shape, but crates.io publishing does not allow ordinary Git dependencies. If `cargo publish --dry-run` is required to pass now, the implementation must either depend on crates.io-published sibling crates or use a workspace/release strategy that makes publish metadata valid.

The LIM-9 canonical link is not specified in issue `#5`. Use the best available repository issue/reference if discoverable during implementation; otherwise include a clear `LIM-9` textual reference and avoid fabricating a URL.

The existing `rust.yml` workflow duplicates the requested CI. Removing it is cleaner, but if repository policy prefers preserving historical workflows, keep it only if it does not duplicate or conflict with `ci.yml`.
