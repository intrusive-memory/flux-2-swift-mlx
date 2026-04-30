# Claude-Specific Agent Instructions

**⚠️ Read [AGENTS.md](AGENTS.md) first.** That file holds the universal project documentation, branching/release flow, testing standard, and critical rules. This file only adds Claude-specific tooling guidance on top.

---

## 1. Build Tools — Never `swift build` or `swift test`

The user's global `~/.claude/CLAUDE.md` forbids `swift build` and `swift test`. This applies in full here. Use one of:

### XcodeBuildMCP (preferred — local development)

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

- No `Makefile` — don't try `make build` / `make test` / `make lint`. The `ship-swift-library` skill assumes one; for this repo, skip those steps.
- No `swift-format` configuration. Don't run formatters speculatively — there is no enforced style and an unconfigured run will produce mass churn.
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

## 5. Claude-Specific Critical Rules

In addition to the universal rules in [AGENTS.md §9](AGENTS.md#9-universal-critical-rules):

1. **Never use `swift build` / `swift test`.** Use XcodeBuildMCP locally; `xcodebuild` in CI.
2. **Always pass `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES`** when invoking `xcodebuild`. MLX is arm64-only.
3. **Don't invent a `make lint` step here.** This repo has no Makefile and no formatter config.
4. **Don't manually upload release tarballs.** No `release.yml` exists; the right answer is to add one in a separate PR, not to hand-build artifacts at release time.

---

**Last updated**: 2026-04-30 (v2.7.0)
