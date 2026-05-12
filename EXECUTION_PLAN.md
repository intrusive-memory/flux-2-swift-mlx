---
mission: flux-2-swift-mlx-instrumentation
feature_name: OPERATION SILICON STETHOSCOPE
source: REQUIREMENTS-instrumentation.md
prior_iteration_brief: docs/incomplete/silicon-stethoscope-01/OPERATION_SILICON_STETHOSCOPE_01_BRIEF.md
host: Vinetas
branch: instrumentation/02
iteration: 2
state: ready
refinement_passes_completed: [atomicity, priority, parallelism, questions, questions-resolved]
hard_discovery_fixes_applied: [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11]
---

# EXECUTION_PLAN.md — flux-2-swift-mlx Instrumentation (Iteration 02)

Produces a `Flux2TelemetryEvent` / `Flux2TelemetryReporter` surface inside `flux-2-swift-mlx` so the Vinetas host can correlate every numerical anomaly (NaN/Inf, gray images, oversaturation, dtype-mismatch artifacts) back to the specific kernel and step that produced it.

This is iteration **02** of the mission. Iteration 01 (`instrumentation/01`, preserved locally) landed 8 of 10 sorties but accumulated 5 build-breaks-at-convergence because the plan deferred all builds to a convergence step. Iteration 02 corrects that structural defect.

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.
> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch.
> **Work Unit** — A grouping of sorties (package, component, phase).

## Cross-repo context

- This plan is the flux-2-swift-mlx slice of `/Users/stovak/Projects/Vinetas/EXECUTION_PLAN.md`.
- Source spec: `REQUIREMENTS-instrumentation.md` (at root; iteration-02-updated with Hard Discovery fixes F1–F11).
- Lessons input: `docs/incomplete/silicon-stethoscope-01/OPERATION_SILICON_STETHOSCOPE_01_BRIEF.md`.
- §6 of the spec (Vinetas-host adapter mapping) is OUT OF SCOPE — it ships in the Vinetas repo.

## Repo constraints (bake into every sortie)

### **CRITICAL — per-sortie compile + test gate (iteration-02 only)**

**Every code-touching sortie ends with both `make build` AND `make test` (or `make test-core`) passing.** No exceptions. Sub-agents run their own builds. The supervising agent does NOT carry a build to a convergence step.

Exit-criteria templates per sortie type:
- **Production code sortie (no new tests):** `make build` succeeds + existing `make test` remains green.
- **Test sortie:** `make build` succeeds + `make test` succeeds with the new test count reported.
- **Refactor / documentation sortie:** `make build` succeeds.

If `make build` fails, the sortie does NOT commit. The agent diagnoses and fixes, OR reports the failure and STOPS without committing. The supervisor never inherits an uncompilable working tree.

### Build tools

- Never `swift build` / `swift test`. Use `make build` / `make test` / `make test-core`. XcodeBuildMCP `swift_package_build` / `swift_package_test` as fallback. Raw `xcodebuild` is CI-only.
- `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES` — non-negotiable (MLX has no x86_64 path).

### Dependencies

- **SwiftTuberia `from: "0.7.0"`** — confirmed latest released tag; mirror the existing `SwiftAcervo` sibling pattern at `Package.swift:57`.
- **swift-tokenizers `from: "0.5.0"`** — already pinned at `Package.swift:56`. No changes.
- All other deps unchanged.

### Branch + release

- All work lands on `instrumentation/02`.
- Sortie 10 cuts a minor release tag for SwiftVinetas to pin to.

### Sendable seam (Hard Discovery F-baseline)

- `Flux2Pipeline` stays `public class … @unchecked Sendable`.
- Telemetry storage uses `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` — NO plain stored property, NO actor migration.
- Hot-path discipline: the denoise-loop `currentTelemetry()` call acquires the lock EXACTLY ONCE per step; multiple stat samples within the step body share the cached pointer.

### `@autoclosure` boundaries (Q2)

The spec uses an `if let telemetry { let stat = … ; await telemetry.capture(...) }` pattern, NOT `@autoclosure`-based deferred evaluation. Every `TuberiaTensorStat.sample` MUST be lexically inside an `if let telemetry` guard — Sortie 6's exit criteria enforce this with grep.

## Iteration-01 brief lessons applied (carry-forward)

| Source | Lesson | How iteration 02 applies it |
|--------|--------|----------------------------|
| Iter-01 brief §1 F1 | `DType.int4` doesn't exist | Sortie 3 task list drops `int4` from dtype bucket examples. |
| Iter-01 brief §1 F2 | `ErrorPhase.imageProcessingFailed` missing | Sortie 1 creates the enum WITH this case (see REQUIREMENTS §3.1). |
| Iter-01 brief §1 F3 | `generationCancelled.stepIndex` should be optional | Sortie 1 creates the case as `stepIndex: Int?` (see REQUIREMENTS §3.1). |
| Iter-01 brief §1 F4 | Anomaly threshold by name | Sortie 5 references `TuberiaTensorStat.defaultOutOfRangeThreshold` directly, never literal. |
| Iter-01 brief §1 F5 | KVExtractStep0 is a single call, not a loop | Sortie 6 emits 3 loop triplets + 1 one-shot triplet = 4 total. |
| Iter-01 brief §1 F6 | 5 denormalize sites; only 2 emit | Sortie 5 uses variable-name discrimination (`finalPatchified` / `patchifiedFinal` emit; `checkpointPatchified` does NOT). |
| Iter-01 brief §1 F7 | Sync init / sync setTimesteps → `Task{}` caveat | Sortie 3 and Sortie 4 use `Task{}` wrapper explicitly; documented in code-comment. |
| Iter-01 brief §1 F8 | `import TestHelpers` required | Sortie 7a + 7b test file templates include the import. |
| Iter-01 brief §1 F9 | `MLXArray.zeros(_:type:)` is static | Sortie 7a task list includes "grep `Sources/...` and `.spm/checkouts/mlx-swift/` for the actual constructor surface before writing tensor allocations." |
| Iter-01 brief §1 F10 | Swift 6 strict static-member access | Sortie 7a + 7b templates require non-static helpers OR explicit `Self.` qualifier. |
| Iter-01 brief §1 F11 | TSan + swift-testing crashes on macOS 26.2 | Sortie 8 writes the lock-contention test in **classic XCTest** (`@objc class … : XCTestCase`), not swift-testing. |
| Iter-01 brief §2 C1 | Sub-agents-skip-builds was a false economy | **Every code-touching sortie runs `make build` + `make test` as part of its exit criteria.** Non-negotiable. |
| Iter-01 brief §2 C2 | "Each touches different files" was unverified | Sortie 3, 4, 5 dispatched **sequentially** (all 3 modify `Flux2Pipeline.swift`). |
| Iter-01 brief §2 C3 | "6 types total" arithmetic error | Sortie 2 exit criterion #3 says **"exactly 7 matches"** (pipeline + 6 subcomponents). |
| Iter-01 brief §2 C4 | Stale line numbers | Sortie tasks reference symbol names (`loadTextEncoder`, `for stepIdx in`). Line numbers in tasks are advisory; dispatch prompts re-audit before launching. |

## Parallelism Structure

**Critical path:** Sortie 1 → Sortie 2 → Sortie 6 → Sortie 7a → Sortie 10 (5 sorties).

**Execution groups:**
- **Group A (Layer 1, foundation):** Sortie 1 — sub-agent. Runs make build + make test before commit.
- **Group B (Layer 2, seam):** Sortie 2 — sub-agent. Runs make build + make test before commit.
- **Group C (Layer 3, non-hot-path emissions):** Sortie 3 → Sortie 4 → Sortie 5 sequentially (NOT parallel — all 3 modify `Flux2Pipeline.swift`; iteration-01 confirmed this in brief decision D2). Each sub-agent runs make build + make test before commit.
- **Group D (Layer 4, hot path):** Sortie 6 — sub-agent. Pre-read line-range targeting. Runs make build + make test before commit.
- **Group E (Layer 5, tests):** Sortie 7a → Sortie 7b sequentially (7b depends on 7a's `MockTelemetryReporter`); Sortie 8 in parallel with 7b (disjoint files: 7b writes 3 new test files, 8 writes 1 new test file + Makefile edit; no overlap).
- **Group F (Layer 5b, overhead test):** Sortie 9 — sub-agent, **must run on ARM64 hardware**, runs alone for timing fidelity. Runs make build + make test before commit.
- **Group G (Layer 6, release):** Sortie 10 — supervising agent.

**Agent constraints:**
- All emission sorties (1, 2, 3, 4, 5, 6) and test sorties (7a, 7b, 8, 9) are sub-agents. **Every sub-agent runs `make build` + `make test` before commit.**
- Sortie 10 (release) is the only supervising-agent-only sortie (handles PR creation, tagging, GitHub release).
- Max 1 concurrent sub-agent in Group C (sequential). Max 2 concurrent in Group E (7b + 8 parallel after 7a).

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| Telemetry types & protocol | `Sources/Flux2Core/Telemetry/` | 1 | 1 | none |
| Pipeline lock seam | `Sources/Flux2Core/{Pipeline,Loading,Scheduler,Transformer}/` | 2 | 2 | Sortie 1 |
| Non-hot-path emissions | `Sources/Flux2Core/{Loading,Scheduler,Pipeline,VAE}/` | 3, 4, 5 | 3 | Sortie 2 (sequential) |
| Hot-path denoise emissions | `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` (denoise loop bodies) | 6 | 4 | Sortie 5 |
| Functional tests | `Tests/Flux2CoreTests/` | 7a, 7b, 8 | 5 | Sortie 6 (for 7a/7b); Sortie 2 (for 8); 7b waits on 7a artifact |
| Overhead test | `Tests/Flux2CoreTests/` | 9 | 5b | Sortie 6 |
| Release | repo root | 10 | 6 | Sorties 7a, 7b, 8, 9 |

Layers gate execution: a sortie in layer N+1 may not dispatch until every sortie in layer ≤N is COMPLETED.

---

### Sortie 1: Add telemetry types and reporter protocol

**Priority**: 7.5 — foundation score 1, dependency depth 9, risk 1.5.

**Agent assignment**: sub-agent. Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Branch `instrumentation/02` is current.
- [ ] `make build` is green on a clean checkout before any changes.
- [ ] `make test` is green before any changes (baseline = 201 tests / 31 suites at branch HEAD).

**Tasks**:
1. Add `sibling("SwiftTuberia", remote: "https://github.com/intrusive-memory/SwiftTuberia", from: "0.7.0")` to the `dependencies:` array in `Package.swift` (mirror the existing `SwiftAcervo` entry). Add `.product(name: "Tuberia", package: "SwiftTuberia")` to the `Flux2Core` target dependencies.
2. Create directory `Sources/Flux2Core/Telemetry/`.
3. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` per REQUIREMENTS §3.1. **Critical:**
   - `case generationCancelled(stepIndex: Int?)` — note the `?` (F3).
   - `ErrorPhase` enum includes `case imageProcessingFailed` (F2).
4. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` per §3.2 (protocol + `NoopFlux2TelemetryReporter`).
5. Imports: `import Tuberia` (Tuberia must be imported here; iteration-01 brief F8 showed this is the right place); `@preconcurrency import MLX`; `import Foundation`. Do NOT redefine `TuberiaTensorStat` locally.

**Exit criteria** (sortie runs all of these before commit):
- [ ] Files `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` and `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` exist.
- [ ] `Package.swift` contains the `SwiftTuberia` sibling entry pinned `from: "0.7.0"` and the `Flux2Core` target depends on `.product(name: "Tuberia", package: "SwiftTuberia")`.
- [ ] `grep -c "case imageProcessingFailed" Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` returns 1 (F2 baked in).
- [ ] `grep -c "case generationCancelled(stepIndex: Int?)" Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` returns 1 (F3 baked in).
- [ ] `grep -R "struct TuberiaTensorStat" Sources/Flux2Core/Telemetry/` returns nothing.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds (no new tests; existing suite still green).**
- [ ] Commit message starts with `sortie 1:`.

---

### Sortie 2: Add `@unchecked Sendable`-safe lock seam and reporter propagation

**Priority**: 9.0 — dependency depth 8, foundation 1, risk 3 (highest single-piece risk per host doc).

**Agent assignment**: sub-agent (opus). Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Sortie 1 COMPLETED (commit ends with `make build` + `make test` green).

**Tasks**:
1. In `Flux2Pipeline.swift`, add `import os.lock` and a private `_telemetryLock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(initialState: nil)`.
2. Add `public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?)` that takes the lock, stores the reporter, AND propagates to every owned subcomponent that has been instantiated at call time: `textEncoder`, `kleinEncoder`, `transformer`, `scheduler`, and `Flux2WeightLoader` static surface. **VAE (`AutoencoderKLFlux2`) is intentionally NOT in the propagation list** (Q3).
3. Add `fileprivate func currentTelemetry() -> (any Flux2TelemetryReporter)?` that reads the lock.
4. Add the same lock + setter + `currentTelemetry()` to **7 types total** (pipeline + 6 owned subcomponents — iter-01 brief C3 noted the plan's "5 subcomponents" arithmetic was wrong; the enumeration is correct):
   - `Flux2Pipeline` (Pipeline/Flux2Pipeline.swift)
   - `KleinTextEncoder` (Loading/KleinTextEncoder.swift)
   - `DevTextEncoder` (Loading/DevTextEncoder.swift)
   - `Flux2TextEncoder` / Mistral (Loading/MistralEncoder.swift)
   - `FlowMatchEulerScheduler` (Scheduler/FlowMatchEulerScheduler.swift)
   - `Flux2WeightLoader` (Loading/WeightLoader.swift) — static surface only
   - top-level transformer class (Transformer/Flux2Transformer.swift)
5. NO emission sites are wired in this sortie.
6. Verify all types still declare `@unchecked Sendable` (lock is the reason we keep the annotation). `Flux2WeightLoader` has no Sendable annotation because it has only static methods — that's correct.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `grep -R "OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>" Sources/Flux2Core/` returns **exactly 7 matches** (F C3-applied: pipeline + 6 subcomponents, VAE excluded).
- [ ] `grep -R "func setTelemetry" Sources/Flux2Core/` returns **exactly 7 matches**.
- [ ] `grep -R "func currentTelemetry()" Sources/Flux2Core/` returns **exactly 7 matches**.
- [ ] `grep -R "setTelemetry\|OSAllocatedUnfairLock" Sources/Flux2Core/VAE/` returns **0 matches** (VAE clean per Q3).
- [ ] `Flux2Pipeline.setTelemetry` body calls `setTelemetry` on each of `textEncoder?`, `kleinEncoder?`, `transformer?`, `scheduler`, and `Flux2WeightLoader` (record line numbers in commit message).
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds (no new tests).**
- [ ] Commit message starts with `sortie 2:` and lists the 7 modified files.

---

### Sortie 3: Wire weight-load, LoRA, init, error emissions

**Priority**: 6.0 — depth 5, foundation 0, risk 2.

**Agent assignment**: sub-agent. Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Sortie 2 COMPLETED.

**Tasks**:
1. In `WeightLoader.swift`, implement a `static func dtypeHistogram(_ params: [String: MLXArray]) -> [String: Int]` builder. Bucket by dtype string. **Use only dtype cases that exist in `MLX.DType`** — DO NOT include `int4` (F1: it doesn't exist; quantized weights pack into wider int dtypes). The `default:` arm handles unknown dtypes via `"\(dtype)"` interpolation.
2. Around `loadTextEncoder`, `loadTransformer`, `loadVAE` (find via grep, ignore plan line numbers — they will drift): emit `weightLoadStart(component:, path:)` before, `weightLoadComplete(component:, paramCount:, dtypeHistogram:, sizeMB:, durationSeconds:)` after.
3. Around `loadLoRA(_:)`: emit `loraLoadStart` / `loraLoadComplete`.
4. In `unloadAllLoRAs()` (iter-01 brief: this IS the deferred-unmerge equivalent path): emit `loraUnmerged(restoredLayerCount:)`.
5. In `Flux2Pipeline.init` near the END of init body: emit `pipelineInit(model:, quantization:, vaeConfig:, memoryOptimization:)` via `Task { await telemetry.capture(...) }`. Add a code-comment: `// F7: init is sync, capture is async. Hosts should call setTelemetry() before the first generation to receive this event.`
6. Add `public func dispose() async` to `Flux2Pipeline`. Body emits `pipelineDispose(model:)` then clears `transformer`, `vae`, `textEncoder`, `kleinEncoder` to nil. Doc-comment: `/// Hosts (Vinetas) should call dispose() before releasing the pipeline. deinit cannot be async, so explicit tear-down is required for pipelineDispose to fire.`
7. Every `throw Flux2Error.…` site in `Flux2Pipeline.swift` — emit `errorThrown(phase:, errorDescription:, stepIndex:)` IMMEDIATELY before the throw. `phase` maps from `Flux2Error.…` to `ErrorPhase.…` (use `.imageProcessingFailed` for `Flux2Error.imageProcessingFailed` — F2). `stepIndex` is `nil` outside denoise loops; the actual `stepIdx` inside denoise loops (will be added in later steps but the throws inside denoise loops here need `stepIndex:` already set correctly).
8. Emission template MUST be: `if let telemetry = currentTelemetry() { let stat = TuberiaTensorStat.sample(…); await telemetry.capture(.…(…)) }`. NO bare `await reporter.capture(...)` outside an `if let` guard.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `grep -nR "telemetry.capture(.weightLoadStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 3 sites.
- [ ] `grep -nR "telemetry.capture(.weightLoadComplete" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns the same count.
- [ ] `grep -c "throw Flux2Error" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` equals `grep -c "telemetry.capture(.errorThrown" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`. (Record both counts in commit message.)
- [ ] `grep -nR "telemetry.capture(.pipelineInit" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 1.
- [ ] `grep -nR "telemetry.capture(.pipelineDispose" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly 1, inside `public func dispose() async`.
- [ ] `grep -nR "deinit" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` — if `deinit` exists, its body must NOT contain `telemetry.capture` or `await` calls.
- [ ] `grep -n "phase: .imageProcessingFailed" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 1 (F2 in use).
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds (existing suite still green).**
- [ ] Commit message starts with `sortie 3:`.

---

### Sortie 4: Wire text-encoder, VLM, and scheduler emissions

**Priority**: 5.5 — depth 5, foundation 0, risk 2.

**Agent assignment**: sub-agent. Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Sortie 3 COMPLETED.

**Tasks**:
1. Around `textEncoder!.encodeWithPrompt(...)`, `textEncoder!.encode(...)`, and `kleinEncoder!.encode(...)` call sites in `generateWithResult` and the I2I branches: emit `textEncoderForwardStart(encoderName:, promptLength:, upsampleRequested:)` before, `textEncoderForwardComplete(encoderName:, finalPromptLength:, embeddingStat:, durationSeconds:)` after.
   - **Encoder name mapping (Q5):** `Flux2TextEncoder` (Mistral) → `"mistral"`; `KleinTextEncoder` (Qwen3) → `"qwen3"`; `TrainingTextEncoder` → `"qwen3-training"`.
2. Around `textEncoder!.describeImagePathsForPrompt(...)` and `upsamplePromptWithImages(...)`: emit `vlmInterpretStart(imageCount:, encoderUsed:)` and `vlmInterpretComplete(descriptionsProduced:, totalDescriptionLength:, durationSeconds:)`.
3. Inside `FlowMatchEulerScheduler.setTimesteps(...)`, AFTER `mu` is computed and `sigmas` populated: emit `schedulerConfigured(numTrainTimesteps:, numInferenceSteps:, shift:, imageSeqLen:, mu:, sigmasHead: Array(sigmas.prefix(5)), sigmasTail: Array(sigmas.suffix(5)))` exactly once per call.
   - **F7:** `setTimesteps` is sync (existing tests depend on this). Use `Task { await telemetry.capture(...) }` to dispatch the async emission. Capture all scalar values into local `let` constants before the closure to avoid data-race concerns.
4. Use the same `if let telemetry = currentTelemetry()` template; sample `embeddingStat` only inside the guard.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `grep -c "telemetry.capture(.textEncoderForwardStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` equals `grep -c "telemetry.capture(.textEncoderForwardComplete" ...` (start/complete pairs).
- [ ] `grep -c "telemetry.capture(.vlmInterpretStart" ...` equals `grep -c "telemetry.capture(.vlmInterpretComplete" ...`.
- [ ] `grep -c "telemetry.capture(.schedulerConfigured" Sources/Flux2Core/Scheduler/FlowMatchEulerScheduler.swift` returns exactly 1.
- [ ] `grep -nR '"mistral"\|"qwen3"\|"qwen3-training"' Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns matches in your new emission sites (record counts in commit message).
- [ ] Sortie 3's emissions unchanged: `grep -c "telemetry.capture(.errorThrown" ...` count is unchanged.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds.**
- [ ] Commit message starts with `sortie 4:`.

---

### Sortie 5: Wire VAE-decode, anomaly, and cancellation emissions

**Priority**: 6.5 — depth 5, foundation 0, risk 2.5.

**Agent assignment**: sub-agent. Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Sortie 4 COMPLETED.

**Tasks**:
1. **Variable-name discrimination (F6):** the 5 `LatentUtils.denormalizeLatentsWithBatchNorm(...)` call sites in `Flux2Pipeline.swift` divide cleanly:
   - Sites where the target variable is named `checkpointPatchified` (3 sites — mid-loop user-facing checkpoint previews): **emit NOTHING.**
   - Sites where the target variable is named `finalPatchified` or `patchifiedFinal` (2 sites — final-decode for T2I and I2I): **emit `vaeBatchNormDenormalize(beforeStat:, afterStat:)`.** Sample both stats inside an `if let telemetry` guard.
2. Before final VAE forward (`vae!.decode(finalLatents)` at the 2 final-decode paths only): emit `vaeDecodeStart(latentStat:, scalingFactor:)`.
3. After `postprocessVAEOutput(decoded)` succeeds at the 2 final-decode paths: emit `vaeDecodeComplete(pixelStat:, outputDims:, durationSeconds:)`.
4. **Implement `Flux2AnomalyDetector` (Q6 — helper lives in flux, NOT in SwiftTuberia):** create `Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` exporting `enum Flux2AnomalyDetector { static func anomalies(in stat: TuberiaTensorStat, checkZeroLatent: Bool = false, expectedDtype: String? = nil) -> [Flux2TelemetryEvent.AnomalyKind] }`. Returns `.nan` when `stat.hasNaN`, `.inf` when `stat.hasInf`, `.outOfRange` when `abs(stat.max) > TuberiaTensorStat.defaultOutOfRangeThreshold || abs(stat.min) > defaultOutOfRangeThreshold` (F4: reference the constant by name), `.zeroLatent` when `checkZeroLatent && abs(stat.mean) < 1e-6 && stat.std < 1e-6`, `.dtypeUnexpected` when `expectedDtype != nil && stat.dtype != expectedDtype`.
5. After every stat-carrying emission this sortie introduces (vaeDecodeStart, vaeBatchNormDenormalize × 2 stats each, vaeDecodeComplete) AND after Sortie 4's 4 `textEncoderForwardComplete` emissions: add a loop `for kind in Flux2AnomalyDetector.anomalies(in: <stat>, checkZeroLatent: <true for latents, false for pixels>) { await telemetry.capture(.numericalAnomaly(phase: "<sourceEvent>", kind: kind, stepIndex: nil, stat: <stat>)) }` inside the same `if let telemetry` block. Total numericalAnomaly emission points: 8 (Sortie 5's own 4 stat-carrying events × variable counts) + 4 (retrofit on Sortie 4's textEncoderForwardComplete emissions) = 12 minimum.
6. At every cancellation check site (sites in `Flux2Pipeline.swift` that call `Task.checkCancellation()` or guard on `Task.isCancelled`): emit `generationCancelled(stepIndex: <nil for pre-loop sites, stepIdx for in-loop sites>)`. **F3:** the case signature is `stepIndex: Int?` — use `nil` for pre-loop sites; do NOT use sentinel values like `0` or `-1`.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` exists with the `anomalies(in:)` helper.
- [ ] `grep -c "telemetry.capture(.vaeBatchNormDenormalize" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns **exactly 2**. Verify by reading the lines: each must be near a `finalPatchified` or `patchifiedFinal` variable, NEVER near `checkpointPatchified`.
- [ ] `grep -c "telemetry.capture(.vaeDecodeStart" ...` returns exactly 2.
- [ ] `grep -c "telemetry.capture(.vaeDecodeComplete" ...` returns exactly 2.
- [ ] `grep -c "telemetry.capture(.numericalAnomaly" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns **at least 12**.
- [ ] `grep -c "telemetry.capture(.generationCancelled" ...` returns at least 1; at least one match has `stepIndex: nil` (F3 in use).
- [ ] `grep -c "TuberiaTensorStat.defaultOutOfRangeThreshold" Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` returns at least 1 (F4 in use — no literal threshold).
- [ ] Sortie 3/4 emissions unchanged.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds.**
- [ ] Commit message starts with `sortie 5:`.

---

### Sortie 6: Wire the HOT-PATH `denoiseStepComplete` and loop-boundary emissions

**Priority**: 8.5 — depth 4, foundation 0, risk 3 (hot path; +1% overhead budget depends on this).

**Agent assignment**: sub-agent (opus). Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Sortie 5 COMPLETED.

**Pre-read targeting (context discipline)**: Read ONLY these regions in `Flux2Pipeline.swift`:
- The `transformer.forwardKVExtract(...)` one-shot site (±25 lines).
- Each of the 3 `for stepIdx in …` loop bodies (±30 lines each).
- Lines around `_telemetryLock` / `currentTelemetry()` declarations (top of class).
- One of Sortie 4's existing `if let telemetry { sample + capture }` blocks (reference for emission style).
- DO NOT read transformer block source. DO NOT read VAE source.

**Tasks**:
1. Identify the 3 `for stepIdx in` loops + 1 `transformer.forwardKVExtract` one-shot by grep. **F5: there are exactly 3 loops + 1 one-shot, NOT 4 loops.** If grep finds different counts, STOP and report.
2. For each of the **3 loops** (`imageToImageKVCached`, `imageToImageFullRecompute`, `textToImage`):
   - Immediately BEFORE the `for stepIdx in …` line: emit `denoiseLoopStart(variant:, totalSteps:, latentShape:, latentDtype:, initialLatentStat:)`.
   - At the TOP of the loop body: add `let telemetry = currentTelemetry()` — **exactly one lock acquisition per step**. All stat samples within the body use this cached optional via `if let telemetry { ... }`.
   - At the BOTTOM of each step body (after `noisePred` is computed AND `scheduler.step` has run): if `let telemetry`, sample `latentBeforeStat`, `noisePredStat`, `latentAfterStat`, then emit `.denoiseStepComplete(variant:, stepIndex:, totalSteps:, sigma:, timestep:, latentBeforeStat:, noisePredStat:, latentAfterStat:, kvCacheLayerCount:, kvCacheHit:, durationSeconds:)`.
   - Immediately after the loop closing brace: emit `denoiseLoopEnd(variant:, totalSteps:, completedSteps:, finalLatentStat:, durationSeconds:)`.
3. For the **KVExtractStep0 one-shot** (single non-loop call to `transformer.forwardKVExtract(...)`): emit a triplet inside a single `if let telemetry` guard:
   - `denoiseLoopStart(variant: .imageToImageKVExtractStep0, totalSteps: 1, latentShape:, latentDtype:, initialLatentStat:)` BEFORE the call.
   - The existing `forwardKVExtract` call runs.
   - `denoiseStepComplete(variant: .imageToImageKVExtractStep0, stepIndex: 0, totalSteps: 1, sigma:, timestep:, latentBeforeStat:, noisePredStat: <stat of noisePred0>, latentAfterStat: <stat of post-extract latent>, kvCacheLayerCount: kvCache.layerCount, kvCacheHit: nil, durationSeconds:)`.
   - `denoiseLoopEnd(variant: .imageToImageKVExtractStep0, totalSteps: 1, completedSteps: 1, finalLatentStat:, durationSeconds:)`.
4. **`kvCacheLayerCount` / `kvCacheHit` policy (Q7):**
   - `textToImage`: both `nil`.
   - `imageToImageKVExtractStep0`: layerCount = `kvCache.layerCount`, hit = `nil`.
   - `imageToImageKVCached`: layerCount = `kvCache.layerCount`, hit = `true` (hardcoded; false-detection is a follow-up).
   - `imageToImageFullRecompute`: both `nil`.
5. **NumericalAnomaly retrofit (iteration-02 fix from brief deviation #1):** Inside each loop's `denoiseStepComplete` emission, AFTER the `capture(.denoiseStepComplete(...))` call but within the same `if let telemetry` block, add the anomaly loop for each of the 3 stats (`latentBeforeStat`, `noisePredStat`, `latentAfterStat`) — `for kind in Flux2AnomalyDetector.anomalies(in: <stat>, checkZeroLatent: true) { await telemetry.capture(.numericalAnomaly(phase: "denoiseStepComplete", kind: kind, stepIndex: stepIdx, stat: <stat>)) }`. Same retrofit on the KVExtractStep0 one-shot.
6. `durationSeconds` per step measured from `Date()` at top of loop body to `Date()` just before `telemetry.capture`. Capture is OUTSIDE the timing window.
7. **DO NOT** call `currentTelemetry()` more than once per step.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `grep -c "telemetry.capture(.denoiseStepComplete" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns **exactly 4** (3 in-loop + 1 one-shot).
- [ ] `grep -c "telemetry.capture(.denoiseLoopStart" ...` returns **exactly 4**.
- [ ] `grep -c "telemetry.capture(.denoiseLoopEnd" ...` returns **exactly 4**.
- [ ] `grep -c "for stepIdx in" ...` returns **exactly 3** (no loops added or lost).
- [ ] `grep -n "transformer\..*forwardKVExtract" ...` returns exactly 1, with the KVExtractStep0 triplet wrapped around it.
- [ ] In each of the 3 loop bodies, `currentTelemetry()` appears EXACTLY ONCE (record line numbers in commit message).
- [ ] Sortie 5's `numericalAnomaly` count increased by ~12 (3 stats × 4 sites = 12 new anomaly emission points).
- [ ] Every `TuberiaTensorStat.sample(` call in `Flux2Pipeline.swift` is lexically inside an `if let telemetry` block.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds.**
- [ ] Commit message starts with `sortie 6:`.

---

### Sortie 7a: MockTelemetryReporter + weight-load + denoise-step contract tests

**Priority**: 4.5 — depth 1, foundation 1.

**Agent assignment**: sub-agent. Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Sortie 6 COMPLETED.

**API surface verification (F9, F10):** BEFORE writing test code:
- `grep -nR "extension MLXArray\|public static func zeros" .spm/checkouts/mlx-swift/Source/MLX/` — confirm `MLXArray.zeros(_:type:)` is the correct API (static method). Iteration-01 brief F9 confirmed: it is.
- The `swift-version 6` setting in `Package.swift` means Swift 6 strict mode applies. Test files MUST use non-static helpers OR `Self.<helper>(...)` at every call site.

**Tasks**:
1. **`Tests/TestHelpers/MockTelemetryReporter.swift`** (F8 — public type in the TestHelpers target):
   ```swift
   import Flux2Core
   public actor MockTelemetryReporter: Flux2TelemetryReporter {
       private var _events: [Flux2TelemetryEvent] = []
       public init() {}
       public func capture(_ event: Flux2TelemetryEvent) async { _events.append(event) }
       public func events() -> [Flux2TelemetryEvent] { _events }
       public func reset() { _events.removeAll() }
   }
   ```
2. **`Tests/Flux2CoreTests/Flux2TelemetryWeightLoadHistogramTests.swift`**: exercise `Flux2WeightLoader.dtypeHistogram` against hand-built `[String: MLXArray]`. Use `MLXArray.zeros(shape, type: T.self)` (F9 — static method form). All private helpers non-static OR `Self.`-qualified (F10). Header includes `import TestHelpers` (F8). At least 4 tests covering: empty input, single dtype, mixed dtypes, scalar tensors.
3. **`Tests/Flux2CoreTests/Flux2TelemetryDenoiseStepTests.swift`**: Option A — pure contract test. Construct synthetic events (`denoiseLoopStart` + 4× `denoiseStepComplete` + `denoiseLoopEnd`), push through `MockTelemetryReporter.capture`, assert event shape, monotone stepIndex 0-3, and latent chaining invariant (`latentAfterStat[N] == latentBeforeStat[N+1]`).

**Exit criteria** (sortie runs all of these before commit):
- [ ] 3 new files exist at the paths above.
- [ ] `grep -n "public actor MockTelemetryReporter" Tests/TestHelpers/MockTelemetryReporter.swift` returns 1.
- [ ] Both Flux2CoreTests files contain `import TestHelpers` (F8).
- [ ] `grep "MLXArray.zeros" Tests/Flux2CoreTests/Flux2TelemetryWeightLoadHistogramTests.swift` returns matches; `grep "MLXArray(zeros" ...` returns nothing (F9).
- [ ] All private helpers in both test files are either non-static or every call site uses `Self.` (F10). Record any exceptions in commit message.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds** with the new tests counted in the suite total (record new total in commit message vs the pre-sortie baseline).
- [ ] Commit message starts with `sortie 7a:`.

---

### Sortie 7b: KV-cache, anomaly, VAE denorm contract tests

**Priority**: 4.0 — depth 1, foundation 0.

**Agent assignment**: sub-agent. Runs `make build` + `make test` before commit.

**Entry criteria**:
- [ ] Sortie 7a COMPLETED. `Tests/TestHelpers/MockTelemetryReporter.swift` exists.

**Tasks** (same pattern as 7a — Option A contract tests, swift-testing, `import TestHelpers`, non-static helpers or `Self.`):
1. `Tests/Flux2CoreTests/Flux2TelemetryKVCacheHitTests.swift` — Klein9BKV cohort: 4 step events; step 0 is `.imageToImageKVExtractStep0` with `kvCacheHit: nil`; steps 1-3 are `.imageToImageKVCached` with `kvCacheHit: true`; all share same `kvCacheLayerCount`. Plus textToImage cohort: 4 steps, all `kvCacheLayerCount: nil` and `kvCacheHit: nil`.
2. `Tests/Flux2CoreTests/Flux2TelemetryAnomalyTests.swift` — 1 synthetic-event pair test (NaN at step 2 fires both `denoiseStepComplete` + `numericalAnomaly(.nan)`) + direct `Flux2AnomalyDetector.anomalies(in:)` unit tests for all 5 `AnomalyKind` branches (.nan, .inf, .outOfRange, .zeroLatent, .dtypeUnexpected) + 1 negative test.
3. `Tests/Flux2CoreTests/Flux2TelemetryVAEDenormalizationTests.swift` — 1 test asserting exactly 1 `vaeBatchNormDenormalize` event and `afterStat.std != beforeStat.std`; 1 test asserting the trio order (vaeDecodeStart → vaeBatchNormDenormalize → vaeDecodeComplete).

**Exit criteria** (sortie runs all of these before commit):
- [ ] 3 new files at the paths above.
- [ ] All 3 files contain `import TestHelpers`.
- [ ] All private helpers either non-static or `Self.`-qualified.
- [ ] `grep "MockTelemetryReporter" Tests/Flux2CoreTests/Flux2Telemetry*Tests.swift | wc -l` returns at least 3 (each new file references the shared mock).
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds** with the new tests counted (record new total).
- [ ] Commit message starts with `sortie 7b:`.

---

### Sortie 8: Lock-contention test (XCTest, NOT swift-testing) + `make test-tsan` target

**Priority**: 5.0 — depth 1, foundation 0, risk 2.5.

**Agent assignment**: sub-agent. Runs `make build` + `make test` + `make test-tsan` before commit.

**Entry criteria**:
- [ ] Sortie 2 COMPLETED (depends on lock seam only; can run in parallel with 7b after 7a).

**Tasks**:
1. **F11 — use CLASSIC XCTEST, not swift-testing.** Iteration-01 brief: swift-testing + TSan + macOS 26.2 SDK crashes xctest at bootstrap. XCTest's older runtime is mature under TSan.
   ```swift
   import XCTest
   @testable import Flux2Core

   final class Flux2TelemetryLockContentionTests: XCTestCase {
       func testConcurrentSetAndGetTelemetry() async { /* ... */ }
       func testHighConcurrencyStress() async { /* ... */ }
       func testLastWriterWins() async { /* ... */ }
   }
   ```
2. Define a local mock fixture inside the test file (do NOT import `MockTelemetryReporter` from TestHelpers; the lock-contention scope is intentionally self-contained).
3. 10 concurrent tasks call `setTelemetry(...)` with toggling reporters (nil, Noop, local mock); concurrent denoise-loop simulation calls `currentTelemetry()`; assert no torn reads.
4. Add `make test-tsan` target to `Makefile` (the one Makefile edit in the campaign):
   ```
   test-tsan: resolve
       xcodebuild test \
           -scheme $(PACKAGE_SCHEME) \
           -destination '$(DESTINATION_MAC)' \
           -skipPackagePluginValidation \
           -enableThreadSanitizer YES \
           -only-testing Flux2CoreTests/Flux2TelemetryLockContentionTests \
           ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
           COMPILER_INDEX_STORE_ENABLE=NO \
           -clonedSourcePackagesDirPath $(SPM_DIR)
   ```
   Update `.PHONY` and `help` block to include `test-tsan`.
5. **Verify `make test-tsan` runs to completion** under TSan. Iteration-01 brief F11 confirmed the platform-issue is specifically swift-testing + TSan; XCTest should not hit the bootstrap crash.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` exists with `import XCTest` (NOT `import Testing`).
- [ ] `grep "final class.*XCTestCase" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 1.
- [ ] `grep -n "^test-tsan" Makefile` returns 1.
- [ ] `grep "enableThreadSanitizer YES" Makefile` returns 1.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds.**
- [ ] **`make test-tsan` runs to completion** — record elapsed time and result in commit message. If TSan reports a real data race, STOP and report; the lock seam has a bug.
- [ ] Commit message starts with `sortie 8:` and includes the `make test-tsan` result.

---

### Sortie 9: Baseline overhead test (ARM64 hardware, runs alone)

**Priority**: 7.0 — depth 1, foundation 0, risk 3 (public contract of the whole instrumentation surface).

**Agent assignment**: sub-agent, **ARM64 hardware required**. Runs `make build` + `make test` (with the new overhead test) before commit.

**Entry criteria**:
- [ ] Sortie 6 COMPLETED.
- [ ] Sorties 7a, 7b, 8 COMPLETED (so the convergence is fully validated before this timing-sensitive run).
- [ ] Agent has access to ARM64 Apple Silicon hardware.

**Tasks**:
1. Add `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift`.
2. **Mocked-transformer rig (Q10):** build a constant-time transformer stub that returns a deterministic pre-allocated `MLXArray` per step. The test measures **telemetry overhead per step**, not MLX kernel time.
3. Run 3 cohorts × 20 iterations × 20 steps:
   - Cohort A: `setTelemetry(nil)` — baseline (no telemetry at all).
   - Cohort B: `setTelemetry(NoopFlux2TelemetryReporter())` — guard passes but capture is a no-op.
   - Cohort C: `setTelemetry(MockTelemetryReporter())` — full event recording.
4. Take wall-clock median per cohort.
5. Assert Cohort B median ≤ Cohort A median × 1.01 (+1% Noop overhead bound).
6. Assert Cohort C median ≤ Cohort A median × 1.05 (+5% Mock overhead bound).
7. Output as a single stdout line: `OVERHEAD_NOOP_PCT=<X> OVERHEAD_MOCK_PCT=<Y>` for Sortie 10's PR description.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift` exists.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds** with the overhead test passing the +1% / +5% bounds.
- [ ] Two clean back-to-back `make test` runs produce `OVERHEAD_NOOP_PCT` values within 0.5 percentage points of each other (record both in commit message).
- [ ] Commit message starts with `sortie 9:` and includes both overhead numbers.

---

### Sortie 10: Release — PR, tag minor version, publish for Vinetas pin

**Priority**: 8.0 — depth 0, foundation 0, risk 2.

**Agent assignment**: supervising agent only.

**Entry criteria**:
- [ ] Sorties 7a, 7b, 8, 9 all COMPLETED.
- [ ] `make build` + `make test` + `make test-tsan` all green on `instrumentation/02`.

**Tasks**:
1. Push `instrumentation/02` and open a PR against `main` of `intrusive-memory/flux-2-swift-mlx`.
2. PR description MUST include:
   - Link to `REQUIREMENTS-instrumentation.md` (root).
   - Link to the iteration-01 brief at `docs/incomplete/silicon-stethoscope-01/OPERATION_SILICON_STETHOSCOPE_01_BRIEF.md`.
   - `OVERHEAD_NOOP_PCT` / `OVERHEAD_MOCK_PCT` from Sortie 9.
   - `make test-tsan` result from Sortie 8.
   - Explicit note: **SwiftTuberia ≥ 0.7.0** is required.
   - Note on `pipelineInit` / `schedulerConfigured` `Task{}` caveat (F7): hosts should call `setTelemetry()` before first generation to receive these events.
3. Wait for CI green; address review feedback.
4. Merge to `main`.
5. Bump to next minor release version (additive change). Tag the merge commit.
6. Create GitHub release. Note in release body: "SwiftVinetas should pin flux-2-swift-mlx ≥ <new-tag>."

**Exit criteria**:
- [ ] PR is merged to default branch.
- [ ] A new minor-version git tag exists on the merge commit.
- [ ] GitHub release is published.
- [ ] PR description contains overhead numbers, TSan result, and SwiftVinetas-pin note.

---

## Resolved Questions

All 11 questions from iteration 01 remain resolved. The Hard Discovery fixes F1-F11 incorporate iteration-01 findings:

| # | Decision | Where it lives |
|---|----------|----------------|
| Q1 | SwiftTuberia v0.7.0 latest released — pin `from: "0.7.0"` | Sortie 1 task 1 |
| Q2 | Explicit `if let telemetry { ... }` guard, no @autoclosure | Every emission sortie |
| Q3 | VAE class gets NO setter; events fire from inside Flux2Pipeline | Sortie 2 task 4 (exit criterion: 0 matches in VAE/) |
| Q4 | Explicit `public func dispose() async`; `deinit` emits nothing | Sortie 3 task 6 |
| Q5 | Encoder-family naming: mistral / qwen3 / qwen3-training | Sortie 4 task 1 |
| Q6 | Anomaly detector helper in flux, not SwiftTuberia | Sortie 5 task 4 |
| Q7 | kvCacheHit policy: nil for t2i / fullRecompute / extractStep0; true for kvCached | Sortie 6 task 4 |
| Q8 | MockTelemetryReporter in `Tests/TestHelpers/`; tests use synthetic events (no real weights) | Sortie 7a |
| Q9 | `make test-tsan` Makefile target, XCTest test file (F11) | Sortie 8 |
| Q10 | Mocked-transformer rig for overhead test | Sortie 9 |
| Q11 | Repo remote: `intrusive-memory/flux-2-swift-mlx`, default `main` | Sortie 10 |

## Summary

| Metric | Value |
|--------|-------|
| Work units | 7 |
| Total sorties | 10 |
| Dependency structure | layered (1 → 2 → 3 → 4 → 5 → 6 → {7a → 7b ∥ 8} → 9 → 10) |
| Critical path length | 5 sorties (1 → 2 → 6 → 7a → 10) |
| Parallel-eligible sets | {7b, 8} after Sortie 7a |
| Agent allocation | All sortie agents are sub-agents except Sortie 10 (supervising) |
| **Per-sortie compile + test gate** | **Non-negotiable. Every code-touching sortie ends with `make build` + `make test` green.** |
| Hard Discovery fixes applied | F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11 |
| Open questions blocking dispatch | 0 |

## Refinement Pass Results

| Pass | Status | Notes vs iteration 01 |
|------|--------|----------------------|
| 1. Atomicity & Testability | PASS | All exit criteria include `make build` + `make test`. Counts in exit criteria match enumerations in tasks (P1 lesson applied). |
| 2. Prioritization | PASS | Critical path unchanged (1 → 2 → 6 → 7a → 10). |
| 3. Parallelism | PASS | Sorties 3/4/5 are explicitly **sequential** (D2 from iteration 01: all 3 modify Flux2Pipeline.swift). Only Sortie 7b ∥ Sortie 8 is genuinely parallel-eligible. |
| 4. Open Questions | PASS | All 11 questions remain resolved from iteration 01. F1-F11 fixes are baked into the plan, not deferred to first-task work. |

**VERDICT**: Plan is refined, all questions resolved, all 11 Hard Discoveries from iteration 01 have explicit fixes baked into sortie definitions, and the per-sortie compile+test gate is the single biggest structural change vs iteration 01. Ready for execution.
