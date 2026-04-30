# Requirements: Migrate from `swift-transformers` to `DePasqualeOrg/swift-tokenizers`

## Context

- Current: `huggingface/swift-transformers` 1.2.0 (resolved), pulled in `Package.swift:20`, products `Hub` + `Transformers` consumed by `FluxTextEncoders` and `Flux2Core`.
- Target: `DePasqualeOrg/swift-tokenizers` 0.4.2 — tokenizer-only fork; **`Hub` was removed** and lives separately at `DePasqualeOrg/swift-hf-api` (module `HFAPI`).
- Toolchain compatibility is fine: this project is Swift 6.2 / macOS 26 / iOS 26; target package needs ≥ 6.1 / macOS 14 / iOS 17.

## R1 — Package manifest changes (`Package.swift`)

- **R1.1** Replace the `swift-transformers` dependency on line 20 with **two** dependencies:
  - `https://github.com/DePasqualeOrg/swift-tokenizers` from `0.4.2`
  - `https://github.com/DePasqualeOrg/swift-hf-api` from `0.2.0` (replacement for the dropped `Hub` product)
- **R1.2** In the `FluxTextEncoders` target (lines 32–33) and `Flux2Core` target (lines 45–46), replace:
  - `.product(name: "Transformers", package: "swift-transformers")` → `.product(name: "Tokenizers", package: "swift-tokenizers")`
  - `.product(name: "Hub", package: "swift-transformers")` → `.product(name: "HFAPI", package: "swift-hf-api")`
- **R1.3** Delete `Package.resolved` and re-resolve to verify no stray transitive `swift-transformers` references remain (`swift-jinja` will still appear — it is also a dep of swift-tokenizers; this is expected).
- **R1.4** Decide on backend trait: default `Swift` backend is fine; **only opt into `Rust` if benchmarked benefit justifies a binary XCFramework dep**. Recommend staying on Swift backend for v1 of the migration.

## R2 — Source code call-site changes

Confirmed import sites (4 files for `Tokenizers`, 1 for `Hub`):

- **R2.1** `Sources/FluxTextEncoders/FluxTextEncoders.swift:14` — `import Tokenizers` stays the same.
- **R2.2** `Sources/FluxTextEncoders/Tokenizer/TekkenTokenizer.swift:14` — `import Tokenizers` stays the same. Update:
  - `tokenizer.decode(tokens: ...)` → `tokenizer.decode(tokenIds: ...)` at line 398 (and any other call sites). **Silent compile break** if missed.
  - `AutoTokenizer.from(modelFolder: …)` → `AutoTokenizer.from(directory: …)` at the call sites around lines 201, 232.
  - `tokenizer.encode(text:)`, `bosTokenId`, `eosTokenId`, `applyChatTemplate(messages:)` — API-compatible, no changes.
- **R2.3** `Sources/FluxTextEncoders/Embeddings/KleinEmbeddingExtractor.swift:16` — same `decode` / `from(modelFolder:)` audit.
- **R2.4** `Sources/FluxTextEncoders/Generation/Qwen3Generator.swift:12` — same audit.
- **R2.5** `Sources/FluxTextEncoders/Loading/TextEncoderModelDownloader.swift:7` — change `import Hub` → `import HFAPI`.
  - Verify `HubApi` type name and `snapshot(from:matching:)` signature in `swift-hf-api` 0.2.0 — confirm method survived the extraction; if renamed, adapt the three call sites at lines 224, 320, 466 and the `HubApi(downloadBase:)` initializer at line 29.
  - Custom directory configuration (lines 33–35) needs an equivalent in the new module.
- **R2.6** Audit `Sources/Flux2Core/Loading/ModelDownloader.swift` (currently modified per git status) for any additional `Hub` usage; apply the same `import` and API changes.

## R3 — Tokenizer protocol conformance

- **R3.1** The `Tokenizer` protocol in 0.4.2 is `Sendable`. Verify any types this project stores as `Tokenizer` properties (e.g., the `tokenizer` properties in `TekkenTokenizer.swift:42` and `KleinEmbeddingExtractor.swift:23`) are usable in the project's actor/concurrency contexts. Likely a no-op given Swift 6.2 strict concurrency is already in play, but confirm at compile time.

## R4 — Tests

Tests must conform to [`TESTING_REQUIREMENTS.md`](../../TESTING_REQUIREMENTS.md) (repo root) — tier placement, framework (Swift Testing), CI gating, and timeout policy.

- **R4.1** `Tests/FluxTextEncodersTests/TokenizerTests.swift` — apply `decode(tokens:)` → `decode(tokenIds:)` rename; verify all assertions still pass.
- **R4.2** `Tests/FluxTextEncodersTests/TextEncoderModelDirectoryTests.swift`, `FluxTextEncodersTests.swift`, `Flux2CoreTests.swift` — re-run after the manifest swap; `swift-hf-api` semantics may differ for offline / cache-only paths.
- **R4.3** Add a focused tokenizer parity test: load the same `tekken.json` / `tokenizer.json` under both old and new packages and assert `encode`/`decode` round-trip equality on a fixture set **before deleting the old dep** to catch any divergence. Drop the test once migration is committed.

## R5 — Data files / on-disk format

- **R5.1** No changes required. `tokenizer.json`, `tekken.json`, `generation_config.json`, `tokenizer_config.json`, and safetensors layouts are unchanged.

## R6 — CI / build verification

- **R6.1** Per global instructions, do **not** use `swift build` / `swift test` locally — verify with XcodeBuildMCP `swift_package_build` + `swift_package_test`, and ensure GitHub Actions workflows (macOS 26, Swift 6.2+) pass.
- **R6.2** Confirm `swift-hf-api`'s minimum platforms don't conflict with `macOS(.v26)` / `iOS(.v26)` declarations.

## R7 — Risks to flag before starting

- **`swift-hf-api` is younger** than the original Hub module (separate package since early 2026). Validate `HubApi.snapshot(from:matching:)` exists with the same semantics — if not, R2.5 grows from a rename to a non-trivial rewrite of `TextEncoderModelDownloader.swift`.
- **`decode(tokens:)` → `decode(tokenIds:)`** is a one-line-per-call-site edit but easy to miss; grep for `decode(tokens` across all targets and tests.
- **Trait selection** under `xcodebuild` requires `TOKENIZERS_BACKEND=…` env var, not `--traits`. Only relevant if R1.4 selects Rust.
- `Sources/Flux2Core/Loading/ModelDownloader.swift` is currently dirty per `git status` — coordinate with whatever in-flight work is happening there before starting the migration branch.

## Suggested execution order

1. R7 risk check on `swift-hf-api` API parity (read `HubApi` source on GitHub) — **gate** before any code edits.
2. R1 manifest swap on a feature branch.
3. R2 + R3 source edits, file by file, compile-clean each step.
4. R4.3 parity test, then R4.1 / R4.2 existing-test sweep.
5. R6 full CI run on the branch.
6. Delete `Package.resolved`, regenerate, commit.

## References

- swift-tokenizers 0.4.2: https://github.com/DePasqualeOrg/swift-tokenizers/tree/0.4.2
- swift-tokenizers migration section: https://github.com/DePasqualeOrg/swift-tokenizers/blob/0.4.2/README.md
- swift-hf-api: https://github.com/DePasqualeOrg/swift-hf-api
- huggingface/swift-transformers (current): https://github.com/huggingface/swift-transformers
