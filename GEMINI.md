# Gemini-Specific Agent Instructions

**⚠️ Read [AGENTS.md](AGENTS.md) first.** That file holds the universal project documentation, branching/release flow, testing standard, and critical rules. This file only adds Gemini-specific tooling guidance on top.

---

## 1. Build Tools — `make` Targets, or Raw `xcodebuild`

The repo ships a `Makefile` that wraps the canonical `xcodebuild` invocations. Prefer `make` targets when they fit:

- `make build` / `make build-ios` — debug builds
- `make install` / `make release` — copy CLIs + Metal bundle to `./bin`
- `make test` / `make test-fte` / `make test-core` / `make test-gpu`
- `make lint` — `swift format` across the tree (first run produces churn; no config file exists)
- `make help` — full list

Gemini does not have access to XcodeBuildMCP, so when finer control is needed, use raw `xcodebuild` directly. The canonical invocation for tests:

```bash
xcodebuild test \
  -scheme Flux2Swift-Package \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation \
  ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=YES \
  COMPILER_INDEX_STORE_ENABLE=NO \
  -clonedSourcePackagesDirPath .spm \
  -only-testing <FluxTextEncodersTests|Flux2CoreTests|Flux2GPUTests>
```

`ARCHS=arm64` and `ONLY_ACTIVE_ARCH=YES` are non-negotiable — MLX has no x86_64 path.

**Do not use `swift build` or `swift test`** for this repository. The project relies on Xcode-driven dependency resolution and scheme configuration; `swift build` paths regularly diverge from `xcodebuild` paths in this codebase.

---

## 2. iOS Simulator (when needed)

Pin an exact OS version, never `OS=latest`:

```
-destination 'platform=iOS Simulator,name=iPhone 17,OS=26.1'
```

Discover available simulators with `xcrun simctl list devices available` if a different version is needed.

---

## 3. What Does Not Exist

- No `.swift-format` configuration. `make lint` runs with default rules — first run will rewrite a lot of source. Review before committing.
- No `release.yml` workflow — only `tests.yml`. Releases are tagged manually following the `ship-swift-library` flow described in [AGENTS.md §4](AGENTS.md#4-branching-and-release-flow).

---

## 4. SwiftAcervo (v0.16.0+) and App Group Configuration

See [AGENTS.md](./AGENTS.md) § *SwiftAcervo dependency (v0.16.0+) — philosophy and App Group configuration*.

The 0.16 release shifts SwiftAcervo's contract: stop poking the filesystem for model state, prefer `Acervo.availability(_:)` / `Acervo.modelFileExists(_:fileName:)` / `Acervo.fetchManifest(for:)` over `FileManager` probes; expect a four-state `ModelAvailability` (`.notAvailable | .downloading | .partial | .available`); CDN-side `CDNManifest` now requires `primaryRepo` and `components` on the wire, so models shipped under `acervo` < 0.16 must be re-published; the `aws` CLI is no longer a runtime dependency. The mechanical 0.14→0.16 bump in this repo touched only `Package.swift`, `Package.resolved`, and `scripts/wu1-bulk-ship.sh` — no source-file changes. UI-layer `.partial` adoption and removal of FileManager probes in `Flux2ModelPaths` / `ModelManager` are deferred per `TODO.md` Groups C/D.

## 4a. Telemetry Chokepoint Convention

When adding or reviewing instrumentation in this repo *or* in any sibling library (`SwiftTuberia`, `Produciesta`, `Vinetas`, etc.), follow the cross-library chokepoint pattern in [AGENTS.md §11](./AGENTS.md#11-telemetry-chokepoint-convention-cross-library). Shared event names (`pipelineInit`, `weightLoadComplete`, `<phase>Complete`, `<loop>LoopStart` / `<loop>LoopEnd`, `numericalAnomaly`, `errorThrown`, `generationCancelled`) and snake_case sink phases (`<lib>_<noun>_<lifecycle>`) let a single Vinetas-style adapter route events from every library uniformly.

This repo's [REQUIREMENTS-instrumentation.md](./REQUIREMENTS-instrumentation.md) is the reference implementation.

Two non-negotiable rules from §11:
- **Boundaries, not internals.** Per-step / per-block / per-attention-head events are deferred until a real anomaly points the agent at the region.
- **`errorThrown` precedes every `throw`.** No exceptions.

## 5. Gemini-Specific Critical Rules

In addition to the universal rules in [AGENTS.md §10](AGENTS.md#10-universal-critical-rules):

1. **Never use `swift build` / `swift test`.** Prefer `make` targets; otherwise `xcodebuild` directly.
2. **Always pass `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES`** when invoking `xcodebuild` directly. (The Makefile already does this.)
3. **Pin iOS Simulator OS version exactly** — `OS=latest` is unreliable.

---

**Last updated**: 2026-05-23 (v3.3.0) — §4 expanded with SwiftAcervo 0.16 philosophy
