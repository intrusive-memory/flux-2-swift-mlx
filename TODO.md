# SwiftAcervo 0.16.0 Upgrade — TODO

## Context

**Current pin:** `SwiftAcervo` 0.14.0 (`Package.resolved` revision `15fd376...`, declared in `Package.swift:67` as `sibling("SwiftAcervo", remote: "https://github.com/intrusive-memory/SwiftAcervo", from: "0.14.0")`).

**Target:** SwiftAcervo 0.16.0.

**Net summary.** This repo never `switch`es over `ModelAvailability`, never uses slug-keyed APIs, never reads `CDNManifest` fields, and never references `AcervoMigration` or `CDNUploader`. The 0.14 → 0.15 leap is a no-op for this consumer (no `aws`-CLI usage, no `CDNUploader` references, no `ship`/`upload` invocations in `Package`/CI). The 0.15 → 0.16 leap is also nearly free: there is no `switch` over `ModelAvailability` to extend, and no test fixtures hand-write `CDNManifest`. The dominant work is the dependency bump itself, plus opportunistic philosophy-cleanup of FileManager-poking code that the 0.16 UPGRADING doc explicitly calls out as anti-pattern (notably `Flux2ModelPaths.findModelPath`, `ModelManager.calculateDirectorySize`, and `Flux2ModelPaths.directorySize`). None of those rewrites is strictly required to *compile* against 0.16, but they are required to honor the 0.16 philosophy and are the only "real" engineering tasks here.

---

## Group A — Dependency pin and resolution (blocking; do first)

- [ ] **Bump `Package.swift:67`** from `from: "0.14.0"` to `from: "0.16.0"` (keep the `sibling(...)` helper; the local-sibling checkout at `../SwiftAcervo` already reports `0.16.0-dev`). File: `/Users/stovak/Projects/flux-2-swift-mlx/Package.swift`.
- [ ] **Refresh `Package.resolved`** so the `swiftacervo` entry pins to a 0.16.0 (or 0.16.x) release tag instead of revision `15fd376158be1c1d3c50a15fb8c31562034c9fc2` / version `0.14.0`. File: `/Users/stovak/Projects/flux-2-swift-mlx/Package.resolved`.
  - Notes: do this via `swift package resolve` after the `Package.swift` bump; do not hand-edit the revision SHA.
- [ ] **No `.github/workflows` change required.** `.github/workflows/tests.yml:8` only sets `ACERVO_APP_GROUP_ID: group.intrusive-memory.models`, which is unrelated to the version bump. Confirm nothing else in CI installs SwiftAcervo by tag.

---

## Group B — Source-breaking API surface (none required, but verify)

This group is intentionally empty after audit. The 0.16.0 source-break risks listed in `UPGRADING.md` are:

1. **New `ModelAvailability.partial` case.** Audit result: zero `switch` statements over `ModelAvailability` in this repo (verified by `grep -rE "case \.(notAvailable|downloading|available|partial)|switch.*availability" --include='*.swift'`). Nothing to extend.
2. **`CDNManifest.primaryRepo` / `components` required on the wire.** Audit result: zero references to `CDNManifest` anywhere in `Sources/` or `Tests/`. No fixtures to update.
3. **`Acervo.listModels()` now filters by validity.** Audit result: zero call sites for `Acervo.listModels`, `modelInfo`, `modelFamilies`, `gcEmptyModelDirectories`. Nothing to handle.
4. **`AcervoMigration` removed (0.14.1).** Audit result: zero references. Nothing to delete.

- [ ] Re-run the four greps above after the dependency bump to confirm no new call sites slipped in via a merge. If any appear, return to this group.

---

## Group C — Adopt `availability(_:)` and `.partial` in app-layer download UI (recommended, not blocking)

The 0.16 philosophy section ("stop poking the filesystem; ask the library") plus the existing `@Published var isDownloading: Bool` flag on `ModelManager` make this the highest-value optional refactor. `ModelManager.swift` currently maintains its own `isDownloading` flag plus six `Acervo.isModelAvailable` probes, which Step 4 of the 0.14 → 0.16 migration explicitly recommends collapsing into a single `await Acervo.availability(repoId)` call returning `.notAvailable | .downloading | .partial | .available`.

- [ ] **`Sources/Flux2App/ViewModels/ModelManager.swift:136-141`** (`refreshDownloadedModels`): replace the `Acervo.isModelAvailable(model.repoId)` + `try? Acervo.modelDirectory(...)` pair with `await Acervo.availability(model.repoId)` returning `.available`, so a `.partial` download is rendered as a repair-row in the UI instead of being silently treated as "not downloaded."
  - Notes: this changes the function to `async`. Either make `refreshDownloadedModels()` async (caller is already `@MainActor` and is invoked from `Task` contexts) or restructure to do the async fetch off-actor and bounce results back. Prefer the `async` rewrite.
- [ ] **`Sources/Flux2App/ViewModels/ModelManager.swift:173-178`** (`refreshDownloadedQwen3Models`): same treatment as above.
- [ ] **`Sources/Flux2App/ViewModels/ModelManager.swift:287` and `:634`** (`deleteQwen3Model`, `deleteModel`): the `guard Acervo.isModelAvailable(repoId) else { return }` short-circuit will silently no-op when the on-disk model is `.partial`. After bump, switch to `let state = await Acervo.availability(repoId); guard state != .notAvailable else { return }` so the destructive path still cleans up partial dirs.
  - Notes: `Acervo.deleteModel(_:)` is synchronous and tolerates a partial directory; the only thing that changes here is the precondition.
- [ ] **`Sources/Flux2App/ViewModels/ModelManager.swift:57-58`** (`@Published var isDownloading`, `downloadProgress`, `downloadMessage`): consider deprecating these in favor of polling `Acervo.availability(repoId)` for the in-progress repo. This is the "replace your own `isDownloading` flag" cleanup from `UPGRADING.md` Step 4. Defer unless the UI is being touched anyway.

---

## Group D — Replace filesystem-poking helpers (recommended; honors 0.16 philosophy)

The 0.16 UPGRADING doc Step 5 calls out four patterns as bugs. All four exist in this repo, all in `Flux2ModelPaths.swift` / `ModelManager.swift`. None of them break compilation against 0.16 — they are correctness/durability bets the library now offers replacements for.

- [ ] **`Sources/Flux2Core/Loading/Flux2ModelPaths.swift:33-56`** (`findModelPath`): the `FileManager.default.fileExists(...config.json...) || ...model_index.json...` probe at lines 48-54 hardcodes filenames, an anti-pattern the doc explicitly lists. Replacement strategy: ask `Acervo.fetchManifest(for: component.repoId)` once at the call site, then use `Acervo.modelDirectory(for: ...)` for the URL. The "deepest directory containing config files" logic is Flux-specific and needs to be preserved because of the `repoSubfolder` mapping (transformer ships under `transformer/qint8/`, VAE under `vae/`), so this is *not* a pure delete — the subfolder-resolution stays, the file-existence probe goes.
  - Notes: `fetchManifest` is `async throws`, which forces `findModelPath` to become `async throws`. Cascades to every caller (`isDownloaded`, `downloadedSize`, `delete`, all the `ModelManager` refresh paths, `Flux2Pipeline.downloadRequiredModels`, `LoRATrainingHelper`). Significant blast radius; consider whether the existing `Acervo.isModelAvailable` strict-on-disk check is good enough and only rewrite `findModelPath` if a UI bug actually surfaces.
- [ ] **`Sources/Flux2Core/Loading/Flux2ModelPaths.swift:122-139`** (`directorySize`): hand-rolled `FileManager.enumerator` walk over a SwiftAcervo-owned model directory. Replacement: iterate `manifest.files` and sum `sizeBytes`. This is exact (manifest-recorded sizes) instead of approximate (whatever happens to be on disk, including `.part` files and crash debris).
  - Notes: requires a cached manifest accessor. The current 0.16 surface offers `fetchManifest(for:)` which round-trips the CDN; for a local size readout you want the locally-cached `.acervo-manifest.json`. If no synchronous local-manifest accessor exists, leave this alone and file a follow-up issue per the doc's "If you find a case the library does not cover, file an issue rather than reaching around" guidance.
- [ ] **`Sources/Flux2App/ViewModels/ModelManager.swift:148-163`** (`calculateDirectorySize`): exact duplicate of `Flux2ModelPaths.directorySize`. Same disposition as above. Either both become manifest-driven or neither does; pick the same outcome.
- [ ] **`Sources/FluxTextEncoders/FluxTextEncoders.swift:184-187`** (Klein config/tokenizer existence probe used only for `print` debug output): replace the `fileManager.fileExists(atPath: "\(modelPath)/config.json")` and `tokenizer.json` literals with `Acervo.modelFileExists(repoId, fileName: "config.json")` etc. Low-priority diagnostics-only cleanup.

---

## Group E — Call sites that DO compile/work unchanged (audit notes only)

These are documented for the implementer so they know not to touch them.

- [ ] `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift:226-235` — `Acervo.ensureAvailable(repoId, files: [], progress: ...)`. Signature unchanged in 0.16. **No change required.**
- [ ] `Sources/Flux2Core/Training/LoRATrainingHelper.swift:484-493` — same `ensureAvailable` shape. **No change required.**
- [ ] `Sources/Flux2App/ViewModels/ModelManager.swift:251-263, :365-377, :414-426, :599-611` — four `ensureAvailable` call sites, all identical signature. **No change required.**
- [ ] `Sources/FluxTextEncoders/FluxTextEncoders.swift:1045-1056` — `ensureAvailable` + `Acervo.modelDirectory(for:)`. **No change required.**
- [ ] `Sources/FluxTextEncoders/Configuration/TextEncoderModelRegistry.swift:402, :411` — `Acervo.isModelAvailable` + `Acervo.modelDirectory(for:)` used in a CLI `print`-status helper. **No change required**; this is a Step 2a "production gate / status readout, keep" case per `UPGRADING.md`.
- [ ] `Sources/Flux2Core/Loading/KleinTextEncoder.swift:146` — `Acervo.isModelAvailable(candidate.repoId)` in a pick-the-downloaded-variant loop. **No change required**; the new strict semantics is what you want here.
- [ ] `Sources/Flux2Core/Loading/Flux2ModelPaths.swift:21, :33, :100, :106` — `isModelAvailable`, `modelDirectory`, `deleteModel(_:)` (synchronous repo-keyed). All signatures unchanged. **No change required** unless Group D is undertaken.
- [ ] `Sources/Flux2App/ViewModels/ModelManager.swift:72` — `Acervo.sharedModelsDirectory`. Unchanged. **No change required.**

---

## Group F — Test targets

- [ ] **No test code references `SwiftAcervo`.** Grep confirms `Tests/` is empty of `import SwiftAcervo`, `Acervo.`, `ModelAvailability`, `CDNManifest`, and `AcervoDownloader`. **No test fixtures to migrate** for the `.partial` case or for the `primaryRepo`/`components` required-field tightening. If tests are added later that synthesize a model directory with only `config.json`, follow `UPGRADING.md` Step 3 (use `isModelConfigPresent`) or Step 3b (seed a `CDNManifest` fixture via `AcervoDownloader.persistManifest`).

---

## Group G — CI / scripts / build glue

- [ ] **`scripts/wu1-bulk-ship.sh:49-52`** asserts `acervo --version >= 0.8.4`. After this bump the project will run against a 0.16.x CLI; raise the floor to `0.16.0` (or higher) so the script fails fast if an operator has a stale `acervo` in PATH. Also note that the 0.15 / 0.16 `ship` flags (`--slug`, `--spec`, `--dry-run`, `--output-dir`, default-orphan-prune behavior) are available; this script does not currently use them and does not need to. If the operator wants to preserve the prior additive-only upload behavior, add `--keep-orphans` to the `acervo ship "${repo}"` invocations at `scripts/wu1-bulk-ship.sh:133, :140`. The decision is operator-policy, not mechanical.
- [ ] **`.github/workflows/tests.yml`**: only `ACERVO_APP_GROUP_ID` is referenced, unrelated to the version bump. **No change required.**

---

## Questions / Risks

1. **`Flux2ModelPaths.findModelPath` async-ification cascade.** Rewriting `findModelPath` to consult `Acervo.fetchManifest` (per Group D first item) makes the function `async throws`, which forces async/throws through `isDownloaded`, `diskSize`, `downloadedSize`, every `ModelManager.refresh*` method, `Flux2Pipeline`'s `missingModels` accessor (not read in this audit), and at least one `LoRATrainingHelper` site. Estimated cost is non-trivial. **Recommendation:** leave `findModelPath` synchronous and FileManager-based for now; document the deviation; revisit if a `.partial` model directory ever produces a real bug. The strict-on-disk `Acervo.isModelAvailable` already guards the "ready to load" gate.
2. **No local-cached-manifest accessor surfaced in 0.16 public API for synchronous size readout.** Group D's `directorySize` rewrite assumes such an accessor exists; the 0.16 file map lists `Acervo+ManifestAccess.swift` with `fetchManifest(for:)` and `fetchManifest(forComponent:)`, both `async throws`. If sync access is required, that is a SwiftAcervo gap; file upstream rather than poke the filesystem. Confirm by reading `/Users/stovak/Projects/SwiftAcervo/Sources/SwiftAcervo/Acervo+ManifestAccess.swift` before committing to the rewrite.
3. **`Flux2ModelPaths.delete` may delete a shared directory.** The repo already documents (lines 94-97) that `.vae(.standard)` and `.transformer(.klein4B_bf16)` both ship under `black-forest-labs/FLUX.2-klein-4B`. `Acervo.deleteModel(repoId)` removes the whole repo directory. This is not a 0.16 regression — same behavior in 0.14 — but the new `.partial` state means a delete of one component may leave the other showing `.partial` (since some manifest files just vanished). **Recommendation:** when Group C is implemented, verify in the UI that deleting `.transformer(.klein4B_bf16)` shows the VAE row flipping to `.notAvailable`, not `.partial`. If the manifest aggregation treats it as partial, the user gets a confusing repair button. May need to special-case shared-directory components on delete.
4. **Slug-keyed APIs are unused; is that correct for multi-component models?** Klein 4B ships transformer + VAE under the same `black-forest-labs/FLUX.2-klein-4B` repo, and Klein-flavored configurations bundle a Qwen3 text encoder (separate `mlx-community/...` repo). The repo addresses each by `org/repo` directly. The 0.16 slug-keyed `availability(slug:url:)` API is designed for exactly this fan-out. **Question:** should the Flux Klein "model" surface a single slug (`flux2-klein-4b`) that aggregates transformer + VAE + text encoder availability, so the UI shows one row per logical model instead of three? This would require a CDN-side spec file (per `acervo ship --spec`) and a `cdnBaseURL` already configured in `ModelRegistry.cdnBaseURL` (line 440). Out of scope for the mechanical 0.14 → 0.16 bump; flagging as a real architectural opportunity 0.16 unlocks.
5. **The `LoRATrainingHelper.swift` "is downloaded?" probe at line 482 uses `Flux2ModelPaths.findModelPath(for:) == nil`.** This is a soft check that returns false for partially-downloaded models, which then re-issues `ensureAvailable` — exactly the right behavior. If Group D's async-ification is deferred, this code keeps working. If Group D ships, this code becomes `async`-cascade collateral. Flag for the implementer.

---

## Group H — Test CI-skip audit for unshipped-model regressions

**Audit question.** Between the moment SwiftAcervo 0.14 → 0.16 merges and the moment the shipping agent re-publishes the 8 stale CDN manifests (see `/Users/stovak/Projects/MODELS-TO-SHIP.md`), any test that drives `Acervo.ensureAvailable(...)` for one of those repos on a fresh CI runner cache will strict-decode-fail on the missing `CDNManifest.primaryRepo` / `components` fields. This group documents whether any such test currently runs in CI.

**Audit result: no AT RISK tests. Every model-loading test is protected by *two* independent mechanisms.** No new skip plumbing is required for the 0.16 bump.

### H.1 Test target inventory

Per `Package.swift` and `ls Tests/`:

| Target | In CI? | Mechanism |
|---|---|---|
| `FluxTextEncodersTests` | Yes | `.github/workflows/tests.yml:50` — `xcodebuild ... -only-testing FluxTextEncodersTests` |
| `Flux2CoreTests` | Yes | `.github/workflows/tests.yml:85` — `xcodebuild ... -only-testing Flux2CoreTests` |
| `Flux2GPUTests` | **No** | Not in `-only-testing` allowlist. Also `make test-gpu` is local-only per AGENTS.md §6 |
| `TestHelpers` | n/a (library target) | Not a test runner |

The CI workflow uses **explicit `-only-testing` allowlist**, so `Flux2GPUTests` is target-excluded before any in-test guard runs. That alone protects it from the regression.

### H.2 Per-target findings

#### `FluxTextEncodersTests` — PROTECTED (no model loads)
- `ConfigurationTests.swift` — pure value-type / Codable round-trips on `MistralTextConfig`. No `Acervo`, no `loadModel`. **PROTECTED** (no network path).
- `CoverageGapTests.swift` — exercises `MockFlux2Pipeline.generate` (in `Tests/TestHelpers/MockFlux2Pipeline.swift`). No real pipeline, no model load. **PROTECTED**.
- `FluxTextEncodersTests.swift` — asserts on the "Model not loaded" *error string* (line 32) without actually loading. Tokenizer + config tests only. **PROTECTED**.
- `GenerationResultTests.swift`, `HiddenStatesConfigTests.swift`, `ImageProcessorTests.swift` — Codable / value-type tests. **PROTECTED**.
- `ModelRegistryTests.swift` — registry enumeration and static-property assertions (`Qwen3Variant.isGated`, `estimatedSizeGB`, etc.). No fetch. Comments at lines 171, 189, 235 explicitly note the test stays static and does *not* go through Acervo. **PROTECTED**.
- `TokenizerTests.swift` — uses `TekkenTokenizer()` with the default in-bundle vocab path (line 17, "Use default tokenizer (no model path)"). No CDN. **PROTECTED**.

#### `Flux2CoreTests` — PROTECTED (pipelines constructed but never `loadModels`d)
- `Flux2CoreTests.swift` — only references `Flux2Pipeline.cgImage(from:)` (a static utility for CGImage decoding, lines 1523, 1602, 1905, 1939, 1947, 1952, 2000). No pipeline construction, no `loadModels`. **PROTECTED**.
- `Flux2ProcessWideTelemetryTests.swift` — constructs `Flux2Pipeline(model: .klein4B, quantization: .minimal)` at lines 46, 71, 97, 131, then only calls `dispose()` / `setTelemetry`. Verified at `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift:153-184` that the init is synchronous, dispatches one telemetry Task, and performs no download. The download path is `loadModels()` → `downloadRequiredModels()` → `Acervo.ensureAvailable` (lines 197-226). None of these tests calls `loadModels`. **PROTECTED**.
- `Flux2TelemetryBoundaryEventsTests.swift` — same pattern (init + telemetry assertions, lines 30, 74, 95). **PROTECTED**.
- `Flux2TelemetryErrorPathTests.swift` — same pattern (line 23). **PROTECTED**.
- `Flux2TelemetryLockContentionTests.swift` — same pattern; in-source comment at line 26 explicitly documents "`Flux2Pipeline.init` is GPU-free — it only wires a scheduler and downloader." **PROTECTED**.
- `Flux2TelemetryAnomalyTests.swift` — no `Flux2Pipeline`/`Acervo`/`loadModels` references at all (grep clean). **PROTECTED**.
- `ImageToImageTrainingTests.swift` — Codable round-trips on `LoRATrainingConfig.ValidationPromptConfig`. No `LoRATrainingHelper` instantiation, no Acervo. **PROTECTED**.
- `TrainingControlTests.swift` — grep clean for `Flux2Pipeline`, `LoRATrainingHelper`, `Acervo`, `loadModel`. **PROTECTED**.

#### `Flux2GPUTests` — PROTECTED via double mechanism (and not in CI anyway)
Belt-and-suspenders coverage:
1. **Target-level**: not in `-only-testing` allowlist in `.github/workflows/tests.yml`. xcodebuild will not select these for execution on CI.
2. **In-test**: every `@Test` function gates on `guard checkGPUPreconditions(minimumBytes: …) else { return }` (Tests/Flux2GPUTests/GPUPreconditions.swift). The guard requires a Metal device and a physical-memory threshold; even if a CI runner ever picked up these tests, GitHub's `macos-26` runners are virtualized and the Metal device check is already documented (in `FluxTextEncodersGPUTests.swift:31` comment) as "No model path on CI — precondition guard exits gracefully."

Files inventoried and verified all `loadModels()` / encoder load callsites are gated:
- `Flux2CoreGPUTests.swift` — 14 `checkGPUPreconditions` guards, every test that calls `pipeline.loadModels()` (lines 24, 39, 63, 137, 198, 269, 295, 357) is preceded by a guard.
- `Flux2CoreModuleTests.swift` — 22+ `checkGPUPreconditions` guards, every test that hits Metal eval is gated.
- `FluxTextEncodersGPUTests.swift` — guards at lines 29, 53, 74, 95.
- `ImageToImageGPUTests.swift` — same pattern.

### H.3 Repo-mapping spot check

For completeness, the would-be download targets if any of the above ever ran on CI:

| Test surface | Repo it would request | In `MODELS-TO-SHIP.md` "confirmed live"? |
|---|---|---|
| `Flux2Pipeline.loadModels()` with `.klein4B` + `.ultraMinimal` (Flux2GPUTests) | `black-forest-labs/FLUX.2-klein-4B` (transformer + VAE) | Yes |
| Text encoder via Mistral path | `lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-{4,6,8}bit` | Yes (all three sizes listed) |
| Text encoder via Qwen3 path | `lmstudio-community/Qwen3-8B-MLX-8bit` | Yes |

No test references a CDN repo that is missing from `MODELS-TO-SHIP.md`'s "confirmed live" section. **No omission flagged.**

### H.4 Conclusion

- **No source-file edits required.** No new `XCTSkip` or env-var gate needs to be added. The existing two-layer protection (xcodebuild `-only-testing` allowlist + `checkGPUPreconditions` guard) is sufficient to keep CI green during the 0.16-bump / re-ship gap.
- **Optional reinforcement (not blocking)**: if a future CI job ever expands the `-only-testing` allowlist to include `Flux2GPUTests`, the in-test `checkGPUPreconditions` guard would still skip every model-loading test on GitHub's virtualized `macos-26` runners (no Metal device). Document this in `TESTING_REQUIREMENTS.md` next time it is touched, but do not change it for this bump.
- **Recommended ordering**: merge the 0.14 → 0.16 dependency bump (Group A) before the shipping agent re-publishes manifests. CI will not regress because no CI-executed test touches the download path. Local `make test-gpu` runs *will* fail on fresh caches during the gap; that is expected and is the operator's responsibility to time after the re-ship completes.
