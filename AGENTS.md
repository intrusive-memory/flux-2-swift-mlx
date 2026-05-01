# flux-2-swift-mlx — AI Agent Instructions

**Audience**: Claude Code, Gemini, Copilot, Cursor, and other AI development assistants.
**Read agent-specific files for tooling specifics**: [CLAUDE.md](CLAUDE.md), [GEMINI.md](GEMINI.md).

This file is the universal source of truth for the project. Agent-specific files extend it; they do not replace it.

---

## 1. Project Overview

`flux-2-swift-mlx` is a native Swift implementation of [Flux.2](https://blackforestlabs.ai/) image-generation models, running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx-swift). This is the `intrusive-memory` fork; the original upstream is `VincentGourbin/flux-2-swift-mlx`.

The package ships two libraries and one demo executable:

| Product | Kind | Purpose |
|---|---|---|
| `FluxTextEncoders` | library | Mistral Small 3.2 + Qwen3 text encoders, Pixtral VLM, FLUX.2 embeddings |
| `Flux2Core` | library | Flux.2 image-generation pipeline (T2I, I2I, multi-image, LoRA, training) |
| `Flux2App` | executable | SwiftUI demo macOS app |

The `Flux2CLI` and `FluxEncodersCLI` executables were removed in v3.0.0 (OPERATION FAREWELL EMBRACE) when the project narrowed to a library-only fork. Downstream consumers integrate via Swift Package Manager. See `README.md` for end-user usage; this file is for agents working on the codebase.

---

## 2. Platforms and Toolchain

- **Swift tools**: 6.2 (declared in `Package.swift`)
- **Platforms**: `macOS .v26`, `iOS .v26`
- **Hardware**: Apple Silicon required (arm64; MLX has no x86_64 path)
- **CI runner**: `macos-26` (never older — Swift 6.2 + macOS 26 SDK assumed throughout)
- **iOS Simulator destination** (when needed): `'platform=iOS Simulator,name=iPhone 17,OS=26.1'` — pin an exact OS version, never `OS=latest`.

---

## 3. Repository Layout

```
Package.swift                    # SPM manifest (single source of truth for products/targets)
Sources/
  FluxTextEncoders/              # Library: text encoders + tokenizers
  Flux2Core/                     # Library: image-generation pipeline + training
  Flux2App/                      # SwiftUI demo app
Tests/
  FluxTextEncodersTests/         # Unit tests (Swift Testing) — CI-safe, no GPU/models
  Flux2CoreTests/                # Config-only tests (Swift Testing) — CI-safe
  Flux2GPUTests/                 # GPU + downloaded-model tests — local-only, not in CI
  TestHelpers/                   # Shared test utilities (TestImage, MockFlux2Pipeline)
.github/workflows/
  tests.yml                      # PR test workflow (only workflow in repo)
docs/                            # User-facing guides + screenshots + mission archives
TESTING_REQUIREMENTS.md          # Authoritative testing standard — read before adding tests
```

**Version strings** are embedded at four call sites:

- `Sources/Flux2Core/Flux2Core.swift` — `Flux2Core.version`
- `Sources/Flux2Core/Training/Training.swift` — `Training.version`
- `Sources/FluxTextEncoders/FluxTextEncoders.swift` — `FluxTextEncoders.version`
- `Sources/FluxTextEncoders/FluxTextEncoders.swift` — `MistralVersion.version` (Mistral wrapper)

All four must move together. The shipped tag is the truth; the embedded string must match the tag at release time.

---

## 4. Branching and Release Flow

- `development` is the integration branch. PRs target `development` for cross-cutting work; the periodic release PR goes from `development → main`.
- `main` is release-only. Each release is a squash commit on `main` followed by an annotated `vX.Y.Z` tag on that squash commit.
- `development` is protected (`allow_force_pushes: false` by default). The release skill temporarily lifts that protection to resync `development` to `main` after a squash merge.

The full release procedure lives in the `ship-swift-library` skill. Agents should not invent ad-hoc release flows — invoke the skill or follow it verbatim.

---

## 5. Build and Test Entry Points

The repo ships a `Makefile` that wraps the canonical `xcodebuild` invocations from §6 / `TESTING_REQUIREMENTS.md`. It is the preferred entry point for routine builds and CI-safe tests; agents that need finer-grained control can call `xcodebuild` directly with the same flags.

| Target | Purpose |
|---|---|
| `make build` | Debug build, macOS arm64 |
| `make build-ios` | Build for iOS Simulator (iPhone 17, OS 26.1) |
| `make install` | Debug build + copy CLIs and `mlx-swift_Cmlx.bundle` to `./bin` |
| `make release` | Release build + copy to `./bin` |
| `make test` | Run both CI-required suites (`FluxTextEncodersTests` + `Flux2CoreTests`) |
| `make test-fte` / `make test-core` | Individual CI-safe suites |
| `make test-gpu` | `Flux2GPUTests` — local-only, requires GPU + downloaded weights |
| `make lint` | `swift format -i` over `Sources/` + `Tests/` (no `.swift-format` config — first run rewrites a lot) |
| `make lint-check` | Same scan, non-mutating (CI-friendly) |
| `make resolve` | Resolve SPM dependencies into `.spm` |
| `make clean` | Remove `./bin`, `.spm`, and `Flux2Swift-*` DerivedData |

Run `make help` for the full list. CI (`.github/workflows/tests.yml`) currently still calls `xcodebuild` directly — keep CI and the Makefile in sync when changing flags.

---

## 6. Testing Standard

Authoritative document: [TESTING_REQUIREMENTS.md](TESTING_REQUIREMENTS.md). Summary:

| Target | CI | Local | Requires |
|---|---|---|---|
| `FluxTextEncodersTests` | Yes | Yes | Nothing — no GPU, no downloads |
| `Flux2CoreTests` | Partial (config only) | Yes | Nothing for config; GPU + weights for inference |
| `Flux2GPUTests` | No | Yes | Apple Silicon, ≥16 GB RAM, downloaded model weights |

**Required CI status checks** (configured in `development` and `main` branch protection):

- `Test FluxTextEncoders (macOS)`
- `Test Flux2Core — Config Only (macOS)`

If you rename a CI job, the branch-protection contexts must be updated in the same change — otherwise PRs become unmergeable. Use:

```bash
gh api --method PUT repos/intrusive-memory/flux-2-swift-mlx/branches/<branch>/protection --input <json>
```

The full test suite (including `Flux2GPUTests`) runs locally only. Anything taking longer than ~5 seconds on CI is either doing I/O it shouldn't or belongs in `Flux2GPUTests`.

Tests use **Swift Testing** (`@Test`, `Issue.record`), not XCTest. The migration was completed in v2.7.0 (OPERATION MARCHING RELAY).

---

## 7. Common Agent Tasks

### Adding a test
1. Decide the target by what the test needs: no GPU/weights → `FluxTextEncodersTests` or `Flux2CoreTests`; GPU/weights → `Flux2GPUTests`.
2. Use Swift Testing syntax (`@Test`, `#expect`, `Issue.record`). Don't add XCTest dependencies.
3. Apply `.timeLimit(...)` to anything that could hang.

### Bumping a dependency
1. Edit `Package.swift` — keep all deps on `.upToNextMajor(from: "X.Y.Z")` form. Never use `.exact`, never use `.path()`.
2. Verify the dependency has a published GitHub release/tag matching what you pin.
3. Resolve and test locally before pushing.

### Bumping the project version
1. Update all three `version` strings in source (see §3).
2. Update README install snippets if they reference a tag.
3. Ship via the `ship-swift-library` skill. The version bump rides with the release PR.

### Touching CI workflows
1. The repo has exactly one workflow: `.github/workflows/tests.yml`. There is **no `release.yml`** — see §8.
2. Pin every `uses:` to the latest major (`@v6`, `@v5`, etc.) — older majors trigger Node 16/20 deprecation warnings.
3. Job names are load-bearing — they're referenced by branch-protection contexts. Don't rename without updating protection.

---

## 8. Build Artifact Distribution

This fork **does not produce release binaries**. Only `tests.yml` exists; there is no `release.yml` to build/upload tarballs or zipped CLI/app bundles.

Consumers integrate via SwiftPM:

```swift
.package(url: "https://github.com/intrusive-memory/flux-2-swift-mlx", .upToNextMajor(from: "3.0.0"))
```

The upstream `VincentGourbin/flux-2-swift-mlx` does publish App and CLI binaries. The README links pointed there historically; they were corrected to this fork's release URLs in v2.7.0, but the fork itself does not currently host binary artifacts. With the CLI executables removed in v3.0.0, the fork is now SwiftPM-only.

---

## 9. Documentation Index

- [README.md](README.md) — End-user / library-consumer entry point
- [TESTING_REQUIREMENTS.md](TESTING_REQUIREMENTS.md) — Authoritative testing standard
- [docs/CLI.md](docs/CLI.md) — `flux2` CLI reference
- [docs/TextEncoders.md](docs/TextEncoders.md) — `FluxTextEncoders` library and CLI
- [docs/LoRA.md](docs/LoRA.md) — Loading LoRA adapters
- [docs/examples/TRAINING_GUIDE.md](docs/examples/TRAINING_GUIDE.md) — LoRA training
- [docs/CustomModelIntegration.md](docs/CustomModelIntegration.md) — Adding custom MLX models
- [docs/Flux2App.md](docs/Flux2App.md) — Demo app guide
- [docs/missions/](docs/missions/) — Active mission supervisor state
- [docs/complete/](docs/complete/) — Archived mission briefs and execution plans

---

## 10. Universal Critical Rules

These apply to every agent. Agent-specific rules live in [CLAUDE.md](CLAUDE.md) / [GEMINI.md](GEMINI.md).

1. **Never commit directly to `main`.** Releases reach `main` only via squash-merged PRs from `development`.
2. **Never delete `development`.** It is long-lived. The release skill resyncs it onto `main` after each release.
3. **Never edit `Package.resolved` by hand.** It is gitignored and regenerated on resolve.
4. **Never add `.path()` dependencies to `Package.swift`.** Every dep must reference a published GitHub release, pinned `.upToNextMajor(from: ...)`.
5. **Keep all three version strings in lockstep** (see §3). A mismatch ships a binary whose self-reported version disagrees with its tag.
6. **Read files before editing.** Don't push speculative diffs.
7. **Don't rename CI job names without updating branch protection contexts.** It deadlocks merges.
8. **Don't add release tarballs by hand to GitHub releases.** The skill explicitly forbids it; if/when a `release.yml` is added, CI owns binary production.
9. **Follow agent-specific instructions** — see [CLAUDE.md](CLAUDE.md) and [GEMINI.md](GEMINI.md).

---

**Last updated**: 2026-05-01 (v3.0.1)
