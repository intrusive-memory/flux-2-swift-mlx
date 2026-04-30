# Gemini-Specific Agent Instructions

**⚠️ Read [AGENTS.md](AGENTS.md) first.** That file holds the universal project documentation, branching/release flow, testing standard, and critical rules. This file only adds Gemini-specific tooling guidance on top.

---

## 1. Build Tools — Use Standard `xcodebuild`

Gemini does not have access to XcodeBuildMCP. Use raw `xcodebuild` directly. The canonical invocation for tests:

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

- No `Makefile` — don't try `make build` / `make test` / `make lint`.
- No `swift-format` configuration. Don't run formatters speculatively.
- No `release.yml` workflow — only `tests.yml`. Releases are tagged manually following the `ship-swift-library` flow described in [AGENTS.md §4](AGENTS.md#4-branching-and-release-flow).

---

## 4. Gemini-Specific Critical Rules

In addition to the universal rules in [AGENTS.md §9](AGENTS.md#9-universal-critical-rules):

1. **Never use `swift build` / `swift test`.** Use `xcodebuild` directly.
2. **Always pass `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES`** when invoking `xcodebuild`.
3. **Pin iOS Simulator OS version exactly** — `OS=latest` is unreliable.

---

**Last updated**: 2026-04-30 (v2.7.0)
