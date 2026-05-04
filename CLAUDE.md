# Claude-Specific Agent Instructions

**⚠️ Read [AGENTS.md](AGENTS.md) first.** That file holds the universal project documentation, branching/release flow, testing standard, and critical rules. This file only adds Claude-specific tooling guidance on top.

---

## 1. Build Tools — Never `swift build` or `swift test`

The user's global `~/.claude/CLAUDE.md` forbids `swift build` and `swift test`. This applies in full here. Order of preference:

### `make` targets (preferred — matches global "prefer Makefiles" rule)

This repo now ships a `Makefile`. Run `make help` to discover targets. The most common are:

- `make build` / `make build-ios` — debug builds for macOS arm64 / iOS Simulator
- `make install` / `make release` — copy CLIs + `mlx-swift_Cmlx.bundle` to `./bin`
- `make test` (CI-required suites) / `make test-gpu` (local-only)
- `make lint` — run `swift format` across the tree (see §6 — first-run produces churn)

The Makefile wraps `xcodebuild` with the canonical flags from `TESTING_REQUIREMENTS.md` (`ARCHS=arm64 ONLY_ACTIVE_ARCH=YES`, etc.).

### XcodeBuildMCP (when you need finer control)

When working interactively, prefer the XcodeBuildMCP tools over raw shell commands. Relevant operations for this repo:

- `swift_package_build` — build SwiftPM products
- `swift_package_test` — run SwiftPM tests
- `build_macos` / `test_macos` — macOS scheme via xcodebuild
- `build_sim` / `test_sim` — iOS Simulator
- `list_sims` — discover available simulator OS versions

The XcodeBuildMCP tool list is discoverable via the MCP itself; the names above are the ones you will most often need.

### Raw `xcodebuild` (CI / when MCP isn't available)

The CI workflow (`.github/workflows/tests.yml`) uses raw `xcodebuild` because XcodeBuildMCP isn't available on GitHub-hosted runners. The canonical invocation matches `TESTING_REQUIREMENTS.md`:

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

`ARCHS=arm64` and `ONLY_ACTIVE_ARCH=YES` are non-negotiable — MLX has no x86_64 path. The user's auto-memory carries a feedback note reinforcing this.

### What does NOT exist here

- No `.swift-format` configuration file. `make lint` runs `swift format` with default rules, so the first run will rewrite a large amount of source. Treat that diff as a one-time style baseline; review it before committing.
- No `release.yml` workflow. Don't reference `make dist` or expect tarball uploads to fire on `release: published`.

---

## 2. iOS Simulator (when needed)

Pin an exact OS version. The user's global guidance:

```
-destination 'platform=iOS Simulator,name=iPhone 17,OS=26.1'
```

`OS=latest` does **not** work in this user's environment. Always pin.

---

## 3. Branch Protection Edits

The `ship-swift-library` skill resyncs `development` onto `main` after each release. That requires temporarily lifting the `allow_force_pushes: false` protection. Use:

```bash
gh api --method PUT repos/intrusive-memory/flux-2-swift-mlx/branches/development/protection --input <json>
```

Always restore protection (`allow_force_pushes: false`) before reporting the release complete.

---

## 4. Global Settings

The user's global Claude instructions live at `~/.claude/CLAUDE.md`. Key load-bearing rules that apply here:

- **Complete candor** — flag ill-advised actions/architectures up front, even if not asked.
- **Secrets and private data** — never echo env vars, API keys, or contents of `.env`/`.p8` files. Never read `~/.claude/@private` without per-operation consent.
- **Swift/Xcode builds** — use XcodeBuildMCP locally, raw `xcodebuild` in CI; never `swift build` / `swift test`.
- **GitHub Actions** — `runs-on: macos-26` minimum, Swift 6.2+, pin iOS Simulator OS version exactly.

---

## 5. App Group Configuration

See [AGENTS.md](./AGENTS.md) § App Group configuration (required).

## 6. Claude-Specific Critical Rules

In addition to the universal rules in [AGENTS.md §10](AGENTS.md#10-universal-critical-rules):

1. **Never use `swift build` / `swift test`.** Use `make` targets first, then XcodeBuildMCP locally; `xcodebuild` in CI.
2. **Always pass `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES`** when invoking `xcodebuild` directly. MLX is arm64-only. (The Makefile already does this for you.)
3. **Don't manually upload release tarballs.** No `release.yml` exists; the right answer is to add one in a separate PR, not to hand-build artifacts at release time.

---

**Last updated**: 2026-05-04 (v3.0.3-dev)
