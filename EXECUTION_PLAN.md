---
mission: flux-2-swift-mlx-instrumentation
feature_name: OPERATION SILICON STETHOSCOPE
source: REQUIREMENTS-instrumentation.md
prior_iteration_brief: docs/incomplete/silicon-stethoscope-02/OPERATION_SILICON_STETHOSCOPE_02_BRIEF.md
host: Vinetas
branch: instrumentation/03
iteration: 3
state: draft
refinement_passes_completed: []
hard_discovery_fixes_applied: [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]
hard_discovery_fixes_dropped: [F11]
tsan_in_scope: false
spec_errata_applied:
  - "pipelineDispose case carries model: String (was bare case in REQUIREMENTS §3.1 line 115; iter-01/02 emission sites need the payload)"
  - "F11 dropped: REQUIREMENTS §F11 prescribes TSan via classic XCTest; iter-02 proved that fails on macOS 26.2 SDK at xctest bootstrap, AND the user has set project policy that TSan is out of scope. OSAllocatedUnfairLock's correctness is established by Apple API guarantees; Swift 6 strict concurrency covers the rest."
  - "Cancellation sites in Flux2Pipeline.swift are guard-based (`guard let transformer = transformer else { throw .generationCancelled }`), NOT `Task.checkCancellation()`. 3 such sites. Iter-02 Hard Discovery 3."
---

# EXECUTION_PLAN.md — flux-2-swift-mlx Instrumentation (Iteration 03)

Produces a `Flux2TelemetryEvent` / `Flux2TelemetryReporter` surface inside `flux-2-swift-mlx` so the Vinetas host can correlate every numerical anomaly (NaN/Inf, gray images, oversaturation, dtype-mismatch artifacts) back to the specific kernel and step that produced it.

This is iteration **03** of the mission. Iteration 02 reached 9/10 sortie completion (commits `85e8ade..ddcc5b0` preserved on `instrumentation/02`) but rolled back because iter-01's F11 hypothesis was wrong — `TSan + xctest + macOS 26.2 SDK` crashes the runner bootstrap regardless of test framework. Post-rollback the user clarified that **TSan was never load-bearing** for this project: `OSAllocatedUnfairLock` has formal Apple API guarantees, and the project's existing Swift-6 strict concurrency + functional contract tests cover the actual correctness claim.

**Iteration 03 drops TSan entirely.** The lock-contention test ships as a plain test (no `-enableThreadSanitizer`, no `make test-tsan` target). Otherwise the plan inherits iter-02's structure — which was 90% correct — and applies the brief's lessons.

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.
> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.
> **Work Unit** — A grouping of sorties (package, component, phase).

## Cross-repo context

- This plan is the flux-2-swift-mlx slice of `/Users/stovak/Projects/Vinetas/EXECUTION_PLAN.md`.
- Source spec: `REQUIREMENTS-instrumentation.md` at repo root.
- Lessons input: `docs/incomplete/silicon-stethoscope-02/OPERATION_SILICON_STETHOSCOPE_02_BRIEF.md` (including the post-rollback addendum that drops TSan).
- §6 of the spec (Vinetas-host adapter mapping) is OUT OF SCOPE — it ships in the Vinetas repo.

## Spec ↔ plan reconciliation (iter-02 carry-forward)

REQUIREMENTS-instrumentation.md currently contains F11 (TSan via classic XCTest) and a bare `case pipelineDispose` in §3.1. This plan supersedes both:

- **F11 (DROP):** Spec says "Sortie 8 TSan test in classic XCTest." Iter-02 evidence: classic XCTest also crashes at xctest bootstrap with `-enableThreadSanitizer YES` on macOS 26.2 SDK. User policy: no TSan. This plan replaces Sortie 8 with a **plain lock-contention behavioral test** — no TSan flag, no Makefile target, runs under the regular `make test`. Carry this erratum into a REQUIREMENTS update before iter-04 if anyone ever runs it.
- **pipelineDispose (FIX):** Spec §3.1 line 115 declares `case pipelineDispose` (bare). The emission sites in §5 imply a model identifier. Iter-02 resolved this by making the case `case pipelineDispose(model: String)`. Iter-03 inherits that resolution.
- **Cancellation pattern (FIX):** Spec §5 says cancellation fires "at every cancellation check site, currently around line 1071." In practice, the existing code uses guard-based patterns (`guard let transformer = transformer else { throw .generationCancelled }`), not `Task.checkCancellation()`. 3 such sites. Iter-03's Sortie 5 task list bakes this in instead of leaving it as a discovery for the agent.

## Repo constraints (bake into every sortie)

### **CRITICAL — per-sortie compile + test gate**

**Every code-touching sortie ends with both `make build` AND `make test` (or `make test-core`) passing.** No exceptions. Sub-agents run their own builds. The supervising agent does NOT carry a build to a convergence step. This was iter-02's headline structural improvement and ships unchanged into iter-03.

Exit-criteria templates per sortie type:
- **Production code sortie (no new tests):** `make build` succeeds + existing `make test` remains green.
- **Test sortie:** `make build` succeeds + `make test` succeeds with the new test count reported.
- **Refactor / documentation sortie:** `make build` succeeds.

If `make build` fails, the sortie does NOT commit. The agent diagnoses and fixes, OR reports the failure and STOPS without committing.

### Build tools

- Never `swift build` / `swift test`. Use `make build` / `make test` / `make test-core`. XcodeBuildMCP `swift_package_build` / `swift_package_test` as fallback. Raw `xcodebuild` is CI-only.
- `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES` — non-negotiable (MLX has no x86_64 path).

### Dependencies

- **SwiftTuberia `from: "0.7.0"`** — confirmed latest released tag; mirror the existing `SwiftAcervo` sibling pattern at `Package.swift:57`.
- **swift-tokenizers `from: "0.5.0"`** — already pinned. No changes.
- All other deps unchanged.

### Branch + release

- All work lands on `instrumentation/03`.
- Sortie 9 (release) cuts our next minor release version (additive change) for SwiftVinetas to pin to.

### Sendable seam

- `Flux2Pipeline` stays `public class … @unchecked Sendable`.
- Telemetry storage uses `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` — NO plain stored property, NO actor migration.
- Hot-path discipline: the denoise-loop `currentTelemetry()` call acquires the lock EXACTLY ONCE per step; multiple stat samples within the step body share the cached pointer.

### `@autoclosure` boundaries (Q2)

The spec uses an `if let telemetry { let stat = … ; await telemetry.capture(...) }` pattern, NOT `@autoclosure`-based deferred evaluation. Every `TuberiaTensorStat.sample` MUST be lexically inside an `if let telemetry` guard.

### No TSan

Confirming for clarity: **iter-03 ships zero TSan-related artifacts.** No `make test-tsan`. No `-enableThreadSanitizer YES`. No standalone TSan executable. No swift-testing-vs-XCTest distinction motivated by TSan. The lock-contention test is plain swift-testing, runs under `make test` in CI, asserts observable behavior (last-writer-wins, no torn reads, no crashes under high concurrency). Lock-correctness claim is delegated to Apple's `OSAllocatedUnfairLock` API guarantees.

## Iteration-02 brief lessons applied (carry-forward)

| Source | Lesson | How iteration 03 applies it |
|--------|--------|----------------------------|
| Iter-02 brief HD1 | F11 hypothesis wrong; xctest+TSan crashes on macOS 26.2 SDK regardless of framework | TSan dropped from scope. Sortie 8 is plain swift-testing lock-contention test. |
| Iter-02 brief HD2 | Spec ↔ plan disagree on `pipelineDispose` signature | Plan uses `case pipelineDispose(model: String)`. Refinement pass adds spec↔plan cross-reference step. |
| Iter-02 brief HD3 | Cancellation pattern is guard-based, not `Task.checkCancellation()` | Sortie 5 task list says guard sites, lists 3 known locations. |
| Iter-02 brief HD4 | Q10 no-work baseline made +1%/+5% bounds structurally impossible | Sortie 7c (overhead test) uses a calibrated CPU-spin stub in EVERY cohort, not just the telemetry ones. |
| Iter-02 brief HD5 | TuberiaTensorStat init order is `(shape, dtype, min, max, mean, std, hasNaN, hasInf)` | Test templates use this order. Refinement pass verifies against source. |
| Iter-02 brief HD6 | `denoiseStepComplete` sigma/timestep are `Float`, not `Double` | Test templates use Float. |
| Iter-02 brief HD7 | Sub-agents reverted supervisor frontmatter edits to `EXECUTION_PLAN.md` | Sortie prompts include "preserve pre-existing unstaged edits" clause. Supervisor commits mission metadata before dispatching Sortie 1. |
| Iter-02 brief PD2.1 | Per-sortie compile+test gate worked — zero convergence build-breaks | Preserved verbatim. |
| Iter-02 brief PD2.5 | Multi-line emission formatting breaks literal-grep exit criteria | Exit criteria use anchored patterns like `\.eventName(` or `eventName(`, never `telemetry.capture(.eventName`. |
| Iter-01 brief §1 F1 | `DType.int4` doesn't exist | Sortie 3 task list drops `int4` from dtype bucket examples. |
| Iter-01 brief §1 F2 | `ErrorPhase.imageProcessingFailed` missing | Sortie 1 creates the enum WITH this case. |
| Iter-01 brief §1 F3 | `generationCancelled.stepIndex` should be optional | Sortie 1 creates the case as `stepIndex: Int?`. |
| Iter-01 brief §1 F4 | Anomaly threshold by name | Sortie 5 references `TuberiaTensorStat.defaultOutOfRangeThreshold` directly. |
| Iter-01 brief §1 F5 | KVExtractStep0 is a single call, not a loop | Sortie 6 emits 3 loop triplets + 1 one-shot triplet = 4 total. |
| Iter-01 brief §1 F6 | 5 denormalize sites; only 2 emit | Sortie 5 uses variable-name discrimination (`finalPatchified`/`patchifiedFinal` emit; `checkpointPatchified` does NOT). |
| Iter-01 brief §1 F7 | Sync init / sync setTimesteps → `Task{}` caveat | Sorties 3 and 4 use `Task{}` wrapper explicitly. |
| Iter-01 brief §1 F8 | `import TestHelpers` required | Sortie 7a + 7b test file templates include the import. |
| Iter-01 brief §1 F9 | `MLXArray.zeros(_:type:)` is static | Sortie 7a task list includes API-surface verification step. |
| Iter-01 brief §1 F10 | Swift 6 strict static-member access | Sortie 7a + 7b templates require non-static helpers OR explicit `Self.` qualifier. |

## Parallelism Structure

**Critical path:** Sortie 1 → Sortie 2 → Sortie 6 → Sortie 7a → Sortie 9 (5 sorties).

**Execution groups:**
- **Group A (Layer 1, foundation):** Sortie 1 — sub-agent. Runs `make build` + `make test` before commit.
- **Group B (Layer 2, seam):** Sortie 2 — sub-agent. Runs `make build` + `make test` before commit.
- **Group C (Layer 3, non-hot-path emissions):** Sortie 3 → Sortie 4 → Sortie 5 sequentially (NOT parallel — all 3 modify `Flux2Pipeline.swift`; iter-01 brief decision D2 + iter-02 confirmation).
- **Group D (Layer 4, hot path):** Sortie 6 — sub-agent. Pre-read line-range targeting.
- **Group E (Layer 5, tests):** Sortie 7a → {Sortie 7b ∥ Sortie 8} after Sortie 7a's `MockTelemetryReporter` is in place. 7b writes 3 contract test files, 8 writes the lock-contention test. Disjoint files.
- **Group F (Layer 5b, overhead test):** Sortie 7c — sub-agent, ARM64 hardware required, runs alone for timing fidelity.
- **Group G (Layer 6, release):** Sortie 9 — supervising agent.

**Agent constraints:**
- Sorties 1-7c are sub-agents.
- Sortie 9 (release) is the only supervising-agent-only sortie.
- Group C is sequential (D2 lesson).
- Group E: 7b and 8 can run in parallel after 7a. Sortie 7c (overhead) runs alone.

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| Telemetry types & protocol | `Sources/Flux2Core/Telemetry/` | 1 | 1 | none |
| Pipeline lock seam | `Sources/Flux2Core/{Pipeline,Loading,Scheduler,Transformer}/` | 2 | 2 | Sortie 1 |
| Non-hot-path emissions | `Sources/Flux2Core/{Loading,Scheduler,Pipeline,VAE}/` | 3, 4, 5 | 3 | Sortie 2 (sequential within layer) |
| Hot-path denoise emissions | `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` (denoise loop bodies) | 6 | 4 | Sortie 5 |
| Functional + contention tests | `Tests/Flux2CoreTests/`, `Tests/TestHelpers/` | 7a, 7b, 8 | 5 | 7a: Sortie 6; 7b: Sortie 7a; 8: Sortie 7a |
| Overhead test | `Tests/Flux2CoreTests/` | 7c | 5b | Sorties 7a, 7b, 8 |
| Release | repo root | 9 | 6 | Sortie 7c |

Layers gate execution: a sortie in layer N+1 may not dispatch until every sortie in layer ≤N is COMPLETED.

---

### Sortie 1: Add telemetry types and reporter protocol

**Agent assignment**: sub-agent.

**Entry criteria**:
- [ ] Branch `instrumentation/03` is current.
- [ ] `make build` is green on a clean checkout before any changes.
- [ ] `make test` is green before any changes (baseline = 201 tests / 31 suites at branch HEAD `16adef2`).

**Tasks**:
1. Add `sibling("SwiftTuberia", remote: "https://github.com/intrusive-memory/SwiftTuberia", from: "0.7.0")` to the `dependencies:` array in `Package.swift` (mirror the existing `SwiftAcervo` entry). Add `.product(name: "Tuberia", package: "SwiftTuberia")` to the `Flux2Core` target dependencies.
2. Create directory `Sources/Flux2Core/Telemetry/`.
3. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` per REQUIREMENTS §3.1. **Critical:**
   - `case generationCancelled(stepIndex: Int?)` — note the `?` (F3).
   - `ErrorPhase` enum includes `case imageProcessingFailed` (F2).
   - `case pipelineDispose(model: String)` — spec erratum, carries model name (iter-02 HD2).
4. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` per §3.2 (protocol + `NoopFlux2TelemetryReporter` struct).
5. Imports: `import Tuberia`; `@preconcurrency import MLX`; `import Foundation`. Do NOT redefine `TuberiaTensorStat` locally.

**Exit criteria** (sortie runs all of these before commit):
- [ ] Files `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` and `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` exist.
- [ ] `Package.swift` contains the `SwiftTuberia` sibling entry pinned `from: "0.7.0"` and the `Flux2Core` target depends on `.product(name: "Tuberia", package: "SwiftTuberia")`.
- [ ] `grep -c "case imageProcessingFailed" Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` returns 1 (F2 baked in).
- [ ] `grep -c "case generationCancelled(stepIndex: Int?)" Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` returns 1 (F3 baked in).
- [ ] `grep -c "case pipelineDispose(model: String)" Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` returns 1 (HD2 erratum applied).
- [ ] `grep -R "struct TuberiaTensorStat" Sources/Flux2Core/Telemetry/` returns nothing.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds (no new tests; existing suite still green).**
- [ ] Commit message starts with `sortie 1:`.

---

### Sortie 2: Add `@unchecked Sendable`-safe lock seam and reporter propagation

**Agent assignment**: sub-agent.

**Entry criteria**:
- [ ] Sortie 1 COMPLETED.

**Tasks**:
1. In `Flux2Pipeline.swift`, add `import os.lock` and a private `_telemetryLock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(initialState: nil)`.
2. Add `public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?)` that takes the lock, stores the reporter, AND propagates to every owned subcomponent that has been instantiated at call time: `textEncoder`, `kleinEncoder`, `transformer`, `scheduler`, and `Flux2WeightLoader` static surface. **VAE (`AutoencoderKLFlux2`) is intentionally NOT in the propagation list (Q3).**
3. Add `fileprivate func currentTelemetry() -> (any Flux2TelemetryReporter)?` that reads the lock.
4. Add the same lock + setter + `currentTelemetry()` to **7 types total** (pipeline + 6 owned subcomponents):
   - `Flux2Pipeline` (Pipeline/Flux2Pipeline.swift)
   - `KleinTextEncoder` (Loading/KleinTextEncoder.swift)
   - `DevTextEncoder` (Loading/DevTextEncoder.swift)
   - `Flux2TextEncoder` / Mistral (Loading/MistralEncoder.swift)
   - `FlowMatchEulerScheduler` (Scheduler/FlowMatchEulerScheduler.swift)
   - `Flux2WeightLoader` (Loading/WeightLoader.swift) — static surface only
   - top-level transformer class (Transformer/Flux2Transformer.swift — contains `Flux2Transformer2DModel`)
5. NO emission sites are wired in this sortie.
6. Verify all types still declare `@unchecked Sendable` (lock is the reason we keep the annotation). `Flux2WeightLoader` has no Sendable annotation because it has only static methods — that's correct.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `grep -R "OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>" Sources/Flux2Core/ | wc -l` returns **exactly 7** (pipeline + 6 subcomponents, VAE excluded).
- [ ] `grep -R "func setTelemetry" Sources/Flux2Core/ | wc -l` returns **exactly 7**.
- [ ] `grep -R "func currentTelemetry()" Sources/Flux2Core/ | wc -l` returns **exactly 7**.
- [ ] `grep -R "setTelemetry\|OSAllocatedUnfairLock" Sources/Flux2Core/VAE/` returns **0 matches** (VAE clean per Q3).
- [ ] `Flux2Pipeline.setTelemetry` body calls `setTelemetry` on each of `textEncoder?`, `kleinEncoder?`, `transformer?`, `scheduler`, and `Flux2WeightLoader` (record line numbers in commit message).
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds (no new tests).**
- [ ] Commit message starts with `sortie 2:` and lists the 7 modified files.

---

### Sortie 3: Wire weight-load, LoRA, init, dispose, error emissions

**Agent assignment**: sub-agent.

**Entry criteria**:
- [ ] Sortie 2 COMPLETED.

**Tasks**:
1. In `WeightLoader.swift`, implement a `static func dtypeHistogram(_ params: [String: MLXArray]) -> [String: Int]` builder. Bucket by dtype string. **Use only dtype cases that exist in `MLX.DType`** — DO NOT include `int4` (F1). The `default:` arm handles unknown dtypes via `"\(dtype)"` interpolation.
2. Around `loadTextEncoder`, `loadTransformer`, `loadVAE` (find via grep, ignore plan line numbers — they drift): emit `weightLoadStart(component:, path:)` before, `weightLoadComplete(component:, paramCount:, dtypeHistogram:, sizeMB:, durationSeconds:)` after.
3. Around `loadLoRA(_:)`: emit `loraLoadStart` / `loraLoadComplete`.
4. In `unloadAllLoRAs()`: emit `loraUnmerged(restoredLayerCount:)`.
5. At the END of `Flux2Pipeline.init` body: emit `pipelineInit(model:, quantization:, vaeConfig:, memoryOptimization:)` via `Task { await telemetry.capture(...) }` (F7). Add code-comment: `// F7: init is sync, capture is async. Hosts should call setTelemetry() before the first generation to receive this event.`
6. Add `public func dispose() async` to `Flux2Pipeline`. Body emits `pipelineDispose(model: <modelString>)` then clears `transformer`, `vae`, `textEncoder`, `kleinEncoder` to nil. **Note `pipelineDispose(model:)` carries the model identifier per HD2 erratum.** Doc-comment: `/// Hosts (Vinetas) should call dispose() before releasing the pipeline. deinit cannot be async, so explicit tear-down is required for pipelineDispose to fire.`
7. Every `throw Flux2Error.…` site in `Flux2Pipeline.swift` (~14 sites; verify with grep) — emit `errorThrown(phase:, errorDescription:, stepIndex:)` IMMEDIATELY before the throw. `phase` maps from `Flux2Error.…` to `ErrorPhase.…` (use `.imageProcessingFailed` for `Flux2Error.imageProcessingFailed` — F2). `stepIndex` is `nil` outside denoise loops; `stepIdx` inside denoise-loop throws.
8. Emission template MUST be: `if let telemetry = currentTelemetry() { ... let stat = TuberiaTensorStat.sample(…); await telemetry.capture(.…(…)) }`. NO bare `await reporter.capture(...)` outside an `if let` guard.

**Exit criteria** (sortie runs all of these before commit, using anchored grep patterns):
- [ ] `grep -c "weightLoadStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift Sources/Flux2Core/Loading/WeightLoader.swift | awk -F: '{s+=$2} END {print s}'` returns ≥ 3 sites total.
- [ ] `grep -c "weightLoadComplete" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift Sources/Flux2Core/Loading/WeightLoader.swift | awk -F: '{s+=$2} END {print s}'` returns the same count as `weightLoadStart`.
- [ ] `grep -c "throw Flux2Error" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` equals `grep -c "errorThrown" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`. Record both counts in commit message.
- [ ] `grep -c "pipelineInit" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 1 emission site.
- [ ] `grep -c "pipelineDispose" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 1 emission site, INSIDE a `public func dispose() async` block.
- [ ] `grep -n "deinit" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` — if `deinit` exists, its body must NOT contain `telemetry.capture` or `await` calls.
- [ ] `grep -c "phase: .imageProcessingFailed" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 1 (F2 in use).
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds (existing suite still green).**
- [ ] Commit message starts with `sortie 3:`.

---

### Sortie 4: Wire text-encoder, VLM, and scheduler emissions

**Agent assignment**: sub-agent.

**Entry criteria**:
- [ ] Sortie 3 COMPLETED.

**Tasks**:
1. Around `textEncoder!.encodeWithPrompt(...)`, `textEncoder!.encode(...)`, and `kleinEncoder!.encode(...)` call sites in `generateWithResult` and the I2I branches: emit `textEncoderForwardStart(encoderName:, promptLength:, upsampleRequested:)` before, `textEncoderForwardComplete(encoderName:, finalPromptLength:, embeddingStat:, durationSeconds:)` after.
   - **Encoder name mapping (Q5):** `Flux2TextEncoder` (Mistral) → `"mistral"`; `KleinTextEncoder` (Qwen3) → `"qwen3"`; `TrainingTextEncoder` → `"qwen3-training"` (only if it's actually called from Flux2Pipeline; iter-02 confirmed it's NOT — skip that mapping unless the agent finds a callsite).
2. Around `textEncoder!.describeImagePathsForPrompt(...)` and `upsamplePromptWithImages(...)`: emit `vlmInterpretStart(imageCount:, encoderUsed:)` and `vlmInterpretComplete(descriptionsProduced:, totalDescriptionLength:, durationSeconds:)`.
3. Inside `FlowMatchEulerScheduler.setTimesteps(...)`, AFTER `mu` is computed and `sigmas` populated: emit `schedulerConfigured(numTrainTimesteps:, numInferenceSteps:, shift:, imageSeqLen:, mu:, sigmasHead: Array(sigmas.prefix(5)), sigmasTail: Array(sigmas.suffix(5)))` exactly once per call.
   - **F7:** `setTimesteps` is sync. Use `Task { await telemetry.capture(...) }`. Capture all scalar values into local `let` constants before the closure to avoid data-race concerns.
4. Use the same `if let telemetry = currentTelemetry()` template; sample `embeddingStat` only inside the guard.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `grep -c "textEncoderForwardStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` equals `grep -c "textEncoderForwardComplete" ...`, both ≥ 3.
- [ ] `grep -c "vlmInterpretStart" ...` equals `grep -c "vlmInterpretComplete" ...`, both ≥ 1.
- [ ] `grep -c "schedulerConfigured" Sources/Flux2Core/Scheduler/FlowMatchEulerScheduler.swift` returns exactly 1.
- [ ] `grep -nE '"mistral"|"qwen3"' Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns matches in new emission sites.
- [ ] Sortie 3's emissions unchanged: `grep -c "errorThrown" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` unchanged from Sortie 3.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds.**
- [ ] Commit message starts with `sortie 4:`.

---

### Sortie 5: Wire VAE-decode, anomaly detector, and cancellation emissions

**Agent assignment**: sub-agent.

**Entry criteria**:
- [ ] Sortie 4 COMPLETED.

**Tasks**:
1. **Variable-name discrimination (F6):** the 5 `LatentUtils.denormalizeLatentsWithBatchNorm(...)` call sites in `Flux2Pipeline.swift` divide cleanly:
   - Sites where the target variable is named `checkpointPatchified` (3 sites — mid-loop user-facing checkpoint previews): **emit NOTHING.**
   - Sites where the target variable is named `finalPatchified` or `patchifiedFinal` (2 sites — final-decode for T2I and I2I): **emit `vaeBatchNormDenormalize(beforeStat:, afterStat:)`.** Sample both stats inside an `if let telemetry` guard.
2. Before final VAE forward (`vae!.decode(finalLatents)` at the 2 final-decode paths only): emit `vaeDecodeStart(latentStat:, scalingFactor:)`.
3. After `postprocessVAEOutput(decoded)` succeeds at the 2 final-decode paths: emit `vaeDecodeComplete(pixelStat:, outputDims:, durationSeconds:)`.
4. **Implement `Flux2AnomalyDetector` (Q6):** create `Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` exporting `enum Flux2AnomalyDetector { static func anomalies(in stat: TuberiaTensorStat, checkZeroLatent: Bool = false, expectedDtype: String? = nil) -> [Flux2TelemetryEvent.AnomalyKind] }`. Returns `.nan` when `stat.hasNaN`, `.inf` when `stat.hasInf`, `.outOfRange` when `abs(stat.max) > TuberiaTensorStat.defaultOutOfRangeThreshold || abs(stat.min) > defaultOutOfRangeThreshold` (F4: reference the constant by name, never literal), `.zeroLatent` when `checkZeroLatent && abs(stat.mean) < 1e-6 && stat.std < 1e-6`, `.dtypeUnexpected` when `expectedDtype != nil && stat.dtype != expectedDtype`.
5. After every stat-carrying emission this sortie introduces (vaeDecodeStart, vaeBatchNormDenormalize × 2 stats each, vaeDecodeComplete) AND after Sortie 4's 4 `textEncoderForwardComplete` emissions: add a loop `for kind in Flux2AnomalyDetector.anomalies(in: <stat>, checkZeroLatent: <true for latents, false for pixels>) { await telemetry.capture(.numericalAnomaly(phase: "<sourceEvent>", kind: kind, stepIndex: nil, stat: <stat>)) }` inside the same `if let telemetry` block. Total numericalAnomaly emission points: at least 12.
6. **Cancellation sites are guard-based, not `Task.checkCancellation()` (iter-02 HD3).** 3 known sites in `Flux2Pipeline.swift` with the pattern `guard let transformer = transformer else { throw Flux2Error.generationCancelled }`:
   - Pre-KV-extract site: emit `generationCancelled(stepIndex: nil)`
   - I2I in-loop site: emit `generationCancelled(stepIndex: stepIdx)`
   - T2I in-loop site: emit `generationCancelled(stepIndex: stepIdx)`
   Emit inside `if let telemetry = currentTelemetry() { await telemetry.capture(...) }` BEFORE the throw. Each site already has a Sortie-3 `errorThrown(.generationCancelled, ...)` emission; preserve that and ADD the `.generationCancelled` emission ahead of it. **F3:** the case signature is `stepIndex: Int?` — never use sentinel values like `0` or `-1`.

**Exit criteria** (sortie runs all of these before commit, using anchored grep patterns):
- [ ] `Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` exists with the `anomalies(in:)` helper.
- [ ] `grep -c '\.vaeBatchNormDenormalize(' Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns **exactly 2** emission sites. Verify by reading the lines: each must be near a `finalPatchified` or `patchifiedFinal` variable, NEVER near `checkpointPatchified`.
- [ ] `grep -c '\.vaeDecodeStart(' ...` returns exactly 2 emission sites.
- [ ] `grep -c '\.vaeDecodeComplete(' ...` returns exactly 2 emission sites.
- [ ] `grep -c '\.numericalAnomaly(' Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 12.
- [ ] `grep -c '\.generationCancelled(' Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least 3 (≥1 with `stepIndex: nil`, F3 in use).
- [ ] `grep -c "TuberiaTensorStat.defaultOutOfRangeThreshold" Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` returns at least 1 (F4 in use).
- [ ] Sortie 3/4 emissions unchanged.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds.**
- [ ] Commit message starts with `sortie 5:`.

---

### Sortie 6: Wire the HOT-PATH `denoiseStepComplete` and loop-boundary emissions

**Agent assignment**: sub-agent (opus — hot path; +1%/+5% overhead budget depends on this).

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
   - At the BOTTOM of each step body (after `noisePred` is computed AND `scheduler.step` has run): if `let telemetry`, sample `latentBeforeStat`, `noisePredStat`, `latentAfterStat`, then emit `.denoiseStepComplete(variant:, stepIndex:, totalSteps:, sigma:, timestep:, latentBeforeStat:, noisePredStat:, latentAfterStat:, kvCacheLayerCount:, kvCacheHit:, durationSeconds:)`. **Note: `sigma` and `timestep` are `Float`, NOT `Double` (iter-02 HD6).**
   - Immediately after the loop closing brace: emit `denoiseLoopEnd(variant:, totalSteps:, completedSteps:, finalLatentStat:, durationSeconds:)`.
3. For the **KVExtractStep0 one-shot** (single non-loop call to `transformer.forwardKVExtract(...)`): emit a triplet of events around the call (denoiseLoopStart with totalSteps:1 + denoiseStepComplete with stepIndex:0/totalSteps:1 + denoiseLoopEnd with totalSteps:1/completedSteps:1). The one-shot is allowed up to 3 `currentTelemetry()` calls (one per event); the once-per-step discipline only applies to in-loop bodies.
4. **`kvCacheLayerCount` / `kvCacheHit` policy (Q7):**
   - `textToImage`: both `nil`.
   - `imageToImageKVExtractStep0`: layerCount = `kvCache.layerCount`, hit = `nil`.
   - `imageToImageKVCached`: layerCount = `kvCache.layerCount`, hit = `true` (hardcoded; false-detection is a follow-up).
   - `imageToImageFullRecompute`: both `nil`.
5. **NumericalAnomaly retrofit:** Inside each loop's `denoiseStepComplete` emission, AFTER the `capture(.denoiseStepComplete(...))` call but within the same `if let telemetry` block, add the anomaly loop for each of the 3 stats (`latentBeforeStat`, `noisePredStat`, `latentAfterStat`) — `for kind in Flux2AnomalyDetector.anomalies(in: <stat>, checkZeroLatent: true) { await telemetry.capture(.numericalAnomaly(phase: "denoiseStepComplete", kind: kind, stepIndex: stepIdx, stat: <stat>)) }`. Same retrofit on the KVExtractStep0 one-shot.
6. **Sortie 3 in-loop pre-throw guards refactor:** Sortie 3 wired errorThrown via `if let telemetry = currentTelemetry() { ... }` inside the existing loop bodies. After this sortie introduces `let telemetry = currentTelemetry()` at the top of each loop body, refactor the in-loop pre-throw guards to reuse the cached `telemetry` binding rather than calling `currentTelemetry()` again. errorThrown emission count must remain unchanged.
7. `durationSeconds` per step measured from `Date()` at top of loop body to `Date()` just before `telemetry.capture`. Capture is OUTSIDE the timing window.
8. **DO NOT** call `currentTelemetry()` more than once per step body (the KVExtractStep0 one-shot's 3-event triplet is an explicit exception, allowed by task 3).

**Exit criteria** (sortie runs all of these before commit, using anchored grep patterns):
- [ ] `grep -c '\.denoiseStepComplete(' Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns **exactly 4** (3 in-loop + 1 one-shot).
- [ ] `grep -c '\.denoiseLoopStart(' ...` returns **exactly 4**.
- [ ] `grep -c '\.denoiseLoopEnd(' ...` returns **exactly 4**.
- [ ] `grep -c "for stepIdx in" ...` returns **exactly 3** (no loops added or lost).
- [ ] `grep -n "transformer\..*forwardKVExtract" ...` returns exactly 1, with the KVExtractStep0 triplet wrapped around it.
- [ ] In each of the 3 loop bodies, `currentTelemetry()` appears EXACTLY ONCE (record line numbers in commit message).
- [ ] Sortie 5's `numericalAnomaly` count increased by ~12 (3 stats × 4 sites = 12 new anomaly emission points).
- [ ] Every `TuberiaTensorStat.sample(` call in `Flux2Pipeline.swift` is lexically inside an `if let telemetry` block.
- [ ] `grep -c "errorThrown" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` unchanged from Sortie 5 baseline.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds.**
- [ ] Commit message starts with `sortie 6:`.

---

### Sortie 7a: MockTelemetryReporter + weight-load + denoise-step contract tests

**Agent assignment**: sub-agent.

**Entry criteria**:
- [ ] Sortie 6 COMPLETED.

**API surface verification (F9, F10)** — BEFORE writing test code:
- `grep -nR "extension MLXArray\|public static func zeros" .spm/checkouts/mlx-swift/Source/MLX/` — confirm `MLXArray.zeros(_:type:)` is the correct API (static method).
- `Package.swift` has `swift-tools-version: 6.2` — Swift 6 strict mode applies. Test files MUST use non-static helpers OR `Self.<helper>(...)` at every call site.
- `TuberiaTensorStat` init order is `(shape, dtype, min, max, mean, std, hasNaN, hasInf)` — verified iter-02 against `/Users/stovak/Projects/SwiftTuberia/Sources/Tuberia/Telemetry/TuberiaTensorStat.swift`. Use this order.

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
2. **`Tests/Flux2CoreTests/Flux2TelemetryWeightLoadHistogramTests.swift`**: exercise `Flux2WeightLoader.dtypeHistogram` against hand-built `[String: MLXArray]`. Use `MLXArray.zeros(shape, type: T.self)` (F9 — static method form). All private helpers non-static OR `Self.`-qualified (F10). Header includes `import TestHelpers` (F8). At least 4 tests: empty input, single dtype, mixed dtypes, scalar tensors.
3. **`Tests/Flux2CoreTests/Flux2TelemetryDenoiseStepTests.swift`**: pure contract test. Construct synthetic events (`denoiseLoopStart` + 4× `denoiseStepComplete` + `denoiseLoopEnd`), push through `MockTelemetryReporter.capture`, assert event shape, monotone stepIndex 0-3, and latent chaining invariant (`latentAfterStat[N] == latentBeforeStat[N+1]`). **Use `Float` for sigma/timestep, not `Double` (HD6).**

**Exit criteria** (sortie runs all of these before commit):
- [ ] 3 new files exist at the paths above.
- [ ] `grep -n "public actor MockTelemetryReporter" Tests/TestHelpers/MockTelemetryReporter.swift` returns 1.
- [ ] Both Flux2CoreTests files contain `import TestHelpers` (F8).
- [ ] `grep "MLXArray.zeros" Tests/Flux2CoreTests/Flux2TelemetryWeightLoadHistogramTests.swift` returns matches; `grep "MLXArray(zeros" ...` returns nothing (F9).
- [ ] All private helpers in both test files are either non-static or every call site uses `Self.` (F10). Record any exceptions in commit message.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds** with the new tests counted (record new total vs the pre-sortie baseline of 201/31).
- [ ] Commit message starts with `sortie 7a:`.

---

### Sortie 7b: KV-cache, anomaly, VAE-denorm contract tests

**Agent assignment**: sub-agent (runs in parallel with Sortie 8 — disjoint files).

**Entry criteria**:
- [ ] Sortie 7a COMPLETED. `Tests/TestHelpers/MockTelemetryReporter.swift` exists.

**Tasks** (same pattern as 7a — swift-testing, `import TestHelpers`, non-static helpers or `Self.`):
1. `Tests/Flux2CoreTests/Flux2TelemetryKVCacheHitTests.swift` — Klein9BKV cohort: 4 step events; step 0 is `.imageToImageKVExtractStep0` with `kvCacheHit: nil`; steps 1-3 are `.imageToImageKVCached` with `kvCacheHit: true`; all share same `kvCacheLayerCount`. Plus textToImage cohort: 4 steps, all `kvCacheLayerCount: nil` and `kvCacheHit: nil`.
2. `Tests/Flux2CoreTests/Flux2TelemetryAnomalyTests.swift` — 1 synthetic-event pair test (NaN at step 2 fires both `denoiseStepComplete` + `numericalAnomaly(.nan)`) + direct `Flux2AnomalyDetector.anomalies(in:)` unit tests for all 5 `AnomalyKind` branches (.nan, .inf, .outOfRange, .zeroLatent, .dtypeUnexpected) + 1 negative test.
3. `Tests/Flux2CoreTests/Flux2TelemetryVAEDenormalizationTests.swift` — 1 test asserting exactly 1 `vaeBatchNormDenormalize` event and `afterStat.std != beforeStat.std`; 1 test asserting the trio order (vaeDecodeStart → vaeBatchNormDenormalize → vaeDecodeComplete).

**Exit criteria** (sortie runs all of these before commit):
- [ ] 3 new files at the paths above.
- [ ] All 3 files contain `import TestHelpers`.
- [ ] All private helpers either non-static or `Self.`-qualified.
- [ ] `grep "MockTelemetryReporter" Tests/Flux2CoreTests/Flux2Telemetry{KVCacheHit,Anomaly,VAEDenormalization}Tests.swift | wc -l` returns at least 3.
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds** with new tests counted.
- [ ] Commit message starts with `sortie 7b:`.

---

### Sortie 8: Lock-contention behavioral test (plain swift-testing — NO TSAN)

**Agent assignment**: sub-agent (runs in parallel with Sortie 7b — disjoint files).

**Entry criteria**:
- [ ] Sortie 7a COMPLETED (for `MockTelemetryReporter` availability if needed; this sortie may also use a local mock).

**Scope clarification (HD1 from iter-02 brief):** This sortie verifies the `OSAllocatedUnfairLock` seam under high concurrent pressure using **observable behavior** — no TSan, no `-enableThreadSanitizer YES`, no `make test-tsan` target, no special Makefile work, no standalone executable. The test runs under the regular `make test` target in CI. Lock-correctness claim is delegated to Apple's `OSAllocatedUnfairLock` API guarantees; Swift 6 strict concurrency covers the surrounding code.

**Tasks**:
1. **`Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift`** — use **swift-testing** (`@Test`, `#expect`, `@Suite`), consistent with the rest of `Flux2CoreTests`.
2. Define a **local mock** inside the test file (do NOT import `MockTelemetryReporter` from TestHelpers; the lock-contention scope is intentionally self-contained).
3. Define a `SeamUnderTest` mirror class with the same lock pattern as `Flux2Pipeline` (`OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` + `withLock` setter/getter). This mirrors the production seam without requiring a full `Flux2Pipeline` instance (which would need model loads).
4. Write 3 tests:
   - **`testConcurrentSetAndGet`** — 100 setTelemetry calls cycling through 10 local mocks + 10 nil-setters + 100 currentTelemetry reads, all in a TaskGroup. Assert no crashes; final state is one of the mocks or nil (no garbage).
   - **`testHighConcurrencyStress`** — 1000 setTelemetry/currentTelemetry interleaved calls under high contention. After the storm, assert setTelemetry visibility (last writer wins).
   - **`testLastWriterWins`** — sequential setTelemetry calls (3 different mocks). Assert `currentTelemetry()` returns the most-recent mock.
5. NO Makefile changes. The tests run under `make test`.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` exists.
- [ ] `grep -c "import Testing" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 1.
- [ ] `grep -c "import XCTest" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 0 (no XCTest; we use swift-testing).
- [ ] `grep -c "OSAllocatedUnfairLock" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns at least 1 (SeamUnderTest mirror).
- [ ] `grep -c "import TestHelpers" Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 0 (local mock per plan).
- [ ] `Makefile` is UNCHANGED — `git diff --stat Makefile` is empty.
- [ ] `grep -c "test-tsan\|enableThreadSanitizer" Makefile Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift` returns 0 (no TSan anywhere).
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds** with the 3 new lock-contention tests counted.
- [ ] Commit message starts with `sortie 8:` and notes "plain swift-testing, no TSan per project policy."

---

### Sortie 7c: Baseline overhead test (ARM64 hardware, runs alone)

**Agent assignment**: sub-agent, **ARM64 hardware required**.

**Entry criteria**:
- [ ] Sortie 6 COMPLETED.
- [ ] Sorties 7a, 7b, 8 COMPLETED.
- [ ] Agent has access to ARM64 Apple Silicon hardware.

**Tasks**:
1. Add `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift`.
2. **Mocked-transformer rig (Q10):** build a constant-time CPU-spin "transformer" stub (~1.9 ms/step or whatever makes the bounds non-trivially measurable). This is the iter-02 HD4 lesson: a no-work baseline makes the +1%/+5% bounds structurally impossible; the stub provides the missing constant work that the production loop has via the real MLX transformer.
3. Use a `SeamUnderTest` mirror (same pattern as Sortie 8 but for the per-step emission shape, not just the lock). Run 3 cohorts × 20 iterations × 20 steps:
   - Cohort A: `setTelemetry(nil)` — baseline (no telemetry at all). Includes the CPU-spin stub.
   - Cohort B: `setTelemetry(NoopFlux2TelemetryReporter())` — guard passes but capture is a no-op. Includes the CPU-spin stub.
   - Cohort C: `setTelemetry(MockTelemetryReporter())` — full event recording. Includes the CPU-spin stub.
4. Take wall-clock median per cohort via `ContinuousClock`.
5. Assert `medianB <= medianA * 1.01` (+1% Noop overhead bound).
6. Assert `medianC <= medianA * 1.05` (+5% Mock overhead bound).
7. Output a single stdout line: `OVERHEAD_NOOP_PCT=<X> OVERHEAD_MOCK_PCT=<Y>` for Sortie 9's PR description.
8. **Scope caveat to document in commit message:** this test measures lock + capture + actor-dispatch overhead. It does NOT measure `TuberiaTensorStat.sample()` per-step cost (the stat is pre-allocated and reused across the timed cohort). This is consistent with Q10's "constant-time transformer stub" intent but worth documenting.
9. **CI flakiness mitigation (iter-02 TEST_CLEANUP_REPORT.md recommendation):** gate the test on a `CI` env var so it does not flake on loaded CI runners:
   ```swift
   @Test("Noop overhead ≤ 1% over baseline; Mock overhead ≤ 5% over baseline")
   func testOverhead() async throws {
       if ProcessInfo.processInfo.environment["CI"] != nil {
           // Timing-sensitive bounds are unreliable on loaded CI runners.
           // The test runs locally for the public-contract claim; CI skips.
           return
       }
       // ... actual test ...
   }
   ```
   Document in the commit message that the test is locally-gated, and that PR descriptions cite the local numbers.

**Exit criteria** (sortie runs all of these before commit):
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift` exists.
- [ ] CI-skip gate is present (verify by grep for `ProcessInfo.processInfo.environment\["CI"\]` or equivalent).
- [ ] **`make build` succeeds.**
- [ ] **`make test` succeeds** — when run locally (no `CI` env var), the overhead bounds pass; when run with `CI=1`, the test is a no-op.
- [ ] Two clean back-to-back local `make test` runs produce `OVERHEAD_NOOP_PCT` values within 0.5 percentage points of each other (record both in commit message).
- [ ] Commit message starts with `sortie 7c:` and includes both overhead numbers + the CI-gate explanation.

---

### Sortie 9: Release — PR, tag next minor release version, publish for Vinetas pin

**Agent assignment**: supervising agent only.

**Entry criteria**:
- [ ] Sorties 7a, 7b, 8, 7c all COMPLETED.
- [ ] `make build` + `make test` green on `instrumentation/03`.

**Tasks**:
1. Push `instrumentation/03` and open a PR against `main` of `intrusive-memory/flux-2-swift-mlx`.
2. PR description MUST include:
   - Link to `REQUIREMENTS-instrumentation.md` (root).
   - Link to the iter-02 brief at `docs/incomplete/silicon-stethoscope-02/OPERATION_SILICON_STETHOSCOPE_02_BRIEF.md` (with the post-rollback addendum that drops TSan).
   - `OVERHEAD_NOOP_PCT` / `OVERHEAD_MOCK_PCT` from Sortie 7c (local-machine numbers; CI skips the test).
   - **Lock-correctness claim**: cites Apple's `OSAllocatedUnfairLock` formal guarantees + Swift 6 strict concurrency + the 3 contention tests from Sortie 8. Explicitly notes that TSan is out of scope per project policy.
   - Explicit note: **SwiftTuberia ≥ 0.7.0** is required.
   - Note on `pipelineInit` / `schedulerConfigured` `Task{}` caveat (F7): hosts should call `setTelemetry()` before first generation to receive these events.
   - Note on `pipelineDispose(model: String)` erratum vs REQUIREMENTS §3.1 line 115 (HD2).
3. Wait for CI green; address review feedback.
4. Merge to `main`.
5. Bump to our next minor release version (additive change). Tag the merge commit.
6. Create GitHub release. Note in release body: "SwiftVinetas should pin flux-2-swift-mlx ≥ <new-tag>."

**Exit criteria**:
- [ ] PR is merged to default branch.
- [ ] A new minor-version git tag exists on the merge commit.
- [ ] GitHub release is published.
- [ ] PR description contains overhead numbers, lock-correctness claim, and SwiftVinetas-pin note.

---

## Resolved Questions

| # | Decision | Where it lives |
|---|----------|----------------|
| Q1 | SwiftTuberia v0.7.0 latest released — pin `from: "0.7.0"` | Sortie 1 task 1 |
| Q2 | Explicit `if let telemetry { ... }` guard, no @autoclosure | Every emission sortie |
| Q3 | VAE class gets NO setter; events fire from inside Flux2Pipeline | Sortie 2 task 4 (exit criterion: 0 matches in VAE/) |
| Q4 | Explicit `public func dispose() async`; `deinit` emits nothing | Sortie 3 task 6 |
| Q5 | Encoder-family naming: mistral / qwen3 (training only if Flux2Pipeline calls it; iter-02 confirmed it doesn't) | Sortie 4 task 1 |
| Q6 | Anomaly detector helper in flux, not SwiftTuberia | Sortie 5 task 4 |
| Q7 | kvCacheHit policy: nil for t2i / fullRecompute / extractStep0; true for kvCached | Sortie 6 task 4 |
| Q8 | MockTelemetryReporter in `Tests/TestHelpers/`; tests use synthetic events (no real weights) | Sortie 7a |
| Q9 | **DROPPED** (was: TSan via classic XCTest). No TSan in iter-03. | Sortie 8 reframed as plain swift-testing lock-contention test |
| Q10 | Mocked-transformer rig with calibrated CPU-spin baseline (HD4) | Sortie 7c |
| Q11 | Repo remote: `intrusive-memory/flux-2-swift-mlx`, default `main` | Sortie 9 |

## Summary

| Metric | Value |
|--------|-------|
| Work units | 7 |
| Total sorties | **9** (vs iter-02's 10 — TSan target dropped) |
| Dependency structure | layered (1 → 2 → 3 → 4 → 5 → 6 → 7a → {7b ∥ 8} → 7c → 9) |
| Critical path length | 5 sorties (1 → 2 → 6 → 7a → 9) |
| Parallel-eligible sets | {7b, 8} after Sortie 7a |
| Agent allocation | All sortie agents are sub-agents except Sortie 9 (supervising) |
| **Per-sortie compile + test gate** | **Non-negotiable. Every code-touching sortie ends with `make build` + `make test` green.** |
| Hard Discovery fixes applied (iter-01) | F1, F2, F3, F4, F5, F6, F7, F8, F9, F10 (F11 DROPPED) |
| Hard Discovery fixes applied (iter-02) | HD1 (TSan drop), HD2 (pipelineDispose payload), HD3 (guard-based cancellation), HD4 (overhead baseline CPU spin), HD5 (TuberiaTensorStat init order), HD6 (Float not Double), HD7 (preserve unstaged supervisor edits) |
| Open questions blocking dispatch | 0 |
