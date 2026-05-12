---
mission: flux-2-swift-mlx-instrumentation
feature_name: OPERATION SILICON STETHOSCOPE
source: REQUIREMENTS-instrumentation.md
host: Vinetas
branch: instrumentation/01
mission_branch: instrumentation/01
starting_point_commit: 3d7d64287f8eaeee2fa71cd545ee5951348c1656
iteration: 1
state: running
refinement_passes_completed: [atomicity, priority, parallelism, questions, questions-resolved]
questions_resolved: 2026-05-12
mission_started: 2026-05-12
---

# EXECUTION_PLAN.md — flux-2-swift-mlx Instrumentation

Produces a `Flux2TelemetryEvent` / `Flux2TelemetryReporter` surface inside `flux-2-swift-mlx` so the Vinetas host can correlate every numerical anomaly (NaN/Inf, gray images, oversaturation, dtype-mismatch artifacts) back to the specific kernel and step that produced it. This is the densest math surface in the Vinetas dep graph; the instrumentation must be additive, non-breaking, and hot-path safe.

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

## Cross-repo context

- This plan is the flux-2-swift-mlx slice of `/Users/stovak/Projects/Vinetas/EXECUTION_PLAN.md`.
- Source spec: `/Users/stovak/Projects/flux-2-swift-mlx/REQUIREMENTS-instrumentation.md`.
- §6 of the spec (Vinetas-host adapter mapping) is OUT OF SCOPE for this plan — it ships in the Vinetas repo.
- §3, §4, §5, §7 of the spec define the sortie boundaries below.

## Repo constraints (bake into every sortie)

- **Builds**: never `swift build` / `swift test`. Prefer `make build` / `make test`; XcodeBuildMCP `swift_package_build` / `swift_package_test` as fallback. Raw `xcodebuild` is CI-only.
- **Dependency floor (RESOLVED)**: SwiftTuberia `v0.7.0` is shipped and publishes `TuberiaTensorStat`. Sortie 1 adds the dependency to `Package.swift` using the existing `sibling(_:remote:from:)` helper (matches the `SwiftAcervo` pattern at `Package.swift:57`). The `Tuberia` product is added to the `Flux2Core` target dependencies.
- **Branch**: all work lands on a single branch `instrumentation/01` against the default branch of flux-2-swift-mlx.
- **Release**: a minor version tag is cut at the end of the campaign so SwiftVinetas can pin to it. The tag is the last sortie's exit criterion.
- **Sendable seam**: `Flux2Pipeline` is `public class … @unchecked Sendable`. Telemetry storage MUST use `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` — no plain stored property, no actor migration.
- **Hot-path discipline**: the denoise-loop `currentTelemetry()` call acquires the lock EXACTLY ONCE per step; multiple stat samples within the step body share the cached pointer.
- **`@autoclosure` boundaries**: see Q2. The spec uses an `if let telemetry { let stat = … ; await telemetry.capture(...) }` pattern, NOT `@autoclosure`-based deferred evaluation. Every `TuberiaTensorStat.sample` MUST be lexically inside an `if let telemetry` guard — Sortie 6's exit criteria enforce this with grep.

## Parallelism Structure

**Critical path**: Sortie 1 → Sortie 2 → Sortie 6 → Sortie 7a → Sortie 10 (5 sorties).

**Parallel execution groups**:
- **Group A (Layer 1, foundation)**: Sortie 1 — supervising agent only (has build step).
- **Group B (Layer 2, seam)**: Sortie 2 — supervising agent only (has build step + propagation across 7 files).
- **Group C (Layer 3, non-hot-path emissions)**: Sortie 3 ∥ Sortie 4 ∥ Sortie 5 — up to 3 sub-agents. Each touches different files. **NO BUILDS** in sub-agents; supervising agent runs the consolidated `make build` after the group converges.
- **Group D (Layer 4, hot path)**: Sortie 6 — supervising agent ONLY. Touches the same file as Sortie 3 (`Flux2Pipeline.swift`) but in different functions; must run AFTER Group C to avoid drift.
- **Group E (Layer 5, tests)**: Sortie 7a ∥ Sortie 7b ∥ Sortie 8 — up to 3 sub-agents writing tests; supervising agent runs `make test`.
- **Group F (Layer 5b, overhead test)**: Sortie 9 — supervising agent ONLY, requires ARM64 hardware and runs alone to avoid wall-clock perturbation.
- **Group G (Layer 6, release)**: Sortie 10 — supervising agent only.

**Agent constraints**:
- **Supervising agent**: handles Sorties 1, 2, 6, 9, 10 and every `make build` / `make test` invocation.
- **Sub-agents (max 3 concurrent)**: handle Sorties 3, 4, 5, 7a, 7b, 8. Sub-agents do not run builds.

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| Telemetry types & protocol | `Sources/Flux2Core/Telemetry/` | 1 | 1 | SwiftTuberia floor bump landed (Q1) |
| Pipeline lock seam | `Sources/Flux2Core/{Pipeline,Loading,Scheduler,Transformer}/` | 2 | 2 | Sortie 1 |
| Non-hot-path emissions | `Sources/Flux2Core/{Loading,Scheduler,Pipeline,VAE}/` | 3, 4, 5 | 3 | Sortie 2 |
| Hot-path denoise emissions | `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` (denoise loop bodies) | 6 | 4 | Sorties 3, 4, 5 |
| Functional tests | `Tests/Flux2CoreTests/` | 7a, 7b, 8 | 5 | Sortie 6 (for 7a/7b); Sortie 2 (for 8) |
| Overhead test | `Tests/Flux2CoreTests/` | 9 | 5b | Sortie 6 |
| Release | repo root | 10 | 6 | Sorties 7a, 7b, 8, 9 |

Layers gate execution: a sortie in layer N+1 may not dispatch until every sortie in layer ≤N is COMPLETED.

---

### Sortie 1: Add telemetry types and reporter protocol

**Priority**: 7.5 — foundation score 1 (used by every later sortie), dependency depth 9 (blocks all others), risk 1.5 (pure type addition).

**Agent assignment**: supervising agent (has build step).

**Entry criteria**:
- [ ] Branch `instrumentation/01` is created off the default branch.
- [ ] `make build` is green on a clean checkout before any changes.

**Tasks**:
1. Add `sibling("SwiftTuberia", remote: "https://github.com/intrusive-memory/SwiftTuberia", from: "0.7.0")` to the `dependencies:` array in `Package.swift` (mirror the existing `SwiftAcervo` entry at `Package.swift:57`). Add `.product(name: "Tuberia", package: "SwiftTuberia")` to the `Flux2Core` target's dependencies (line ~84). `FluxTextEncoders` does NOT need it.
2. Create directory `Sources/Flux2Core/Telemetry/`.
3. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` containing the public `Flux2TelemetryEvent: Sendable` enum and its nested types (`QuantizationManifest`, `WeightComponent`, `DenoiseVariant`, `AnomalyKind`, `ErrorPhase`) exactly as specified in REQUIREMENTS §3.1.
4. Create `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` containing `public protocol Flux2TelemetryReporter: Sendable { func capture(_ event: Flux2TelemetryEvent) async }` and `public struct NoopFlux2TelemetryReporter: Flux2TelemetryReporter` per REQUIREMENTS §3.2.
5. `import Tuberia` for `TuberiaTensorStat`; `@preconcurrency import MLX`; `import Foundation`. Do not redefine `TuberiaTensorStat` locally.
6. No call sites in `Flux2Pipeline` change in this sortie — only the new files are added.

**Context budget**: ~12 turns (R=1 spec, C=2 files, M=0, B=1 `make build`, L≈300, V=4 greps + build). Right-sized.

**Exit criteria**:
- [ ] Files `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` and `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` exist.
- [ ] `Package.swift` contains the `SwiftTuberia` sibling entry pinned `from: "0.7.0"`, and the `Flux2Core` target depends on `.product(name: "Tuberia", package: "SwiftTuberia")`.
- [ ] `make build` succeeds.
- [ ] `make test` succeeds (no new tests, existing suite still green).
- [ ] `grep -R "TuberiaTensorStat" Sources/Flux2Core/Telemetry/` returns matches; `grep -R "struct TuberiaTensorStat" Sources/Flux2Core/Telemetry/` returns nothing (no local redefinition).
- [ ] `grep -R "Flux2TelemetryReporter" Sources/ | grep -v "Telemetry/"` returns empty (no callers exist yet).

---

### Sortie 2: Add `@unchecked Sendable`-safe lock seam and reporter propagation

**Priority**: 9.0 — dependency depth 8, foundation 1, risk 3 (the riskiest single piece in the campaign per host doc risk #1; `@unchecked Sendable` is famously easy to misuse).

**Agent assignment**: supervising agent (has build step + touches 7 files).

**Entry criteria**:
- [ ] Sortie 1 exit criteria all met (telemetry types compile).

**Tasks**:
1. In `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` (line ~72, the class declaration), add `import os.lock` and a private `_telemetryLock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(initialState: nil)`.
2. Add `public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?)` that takes the lock, stores the reporter, AND propagates to every owned subcomponent that has been instantiated at call time: `textEncoder`, `kleinEncoder`, `transformer`, `scheduler`, and `Flux2WeightLoader` static surface (or instance, depending on §5 wiring). **`AutoencoderKLFlux2` (VAE) is intentionally NOT in the propagation list** (Q3 resolved): every VAE-related event in REQUIREMENTS §5 fires from inside `Flux2Pipeline` around VAE calls, not from inside the VAE class.
3. Add `fileprivate func currentTelemetry() -> (any Flux2TelemetryReporter)?` that reads the lock.
4. Add the same lock + setter + `currentTelemetry()` to each of: `KleinTextEncoder` (`KleinTextEncoder.swift:22`), `DevTextEncoder` (`DevTextEncoder.swift:15`), `Flux2TextEncoder` / Mistral (`Loading/MistralEncoder.swift`), `FlowMatchEulerScheduler` (`FlowMatchEulerScheduler.swift:34`), `Flux2WeightLoader` (`WeightLoader.swift:9`), and the top-level transformer class (`Transformer/Flux2Transformer.swift`). **6 types total** (pipeline + 5 owned subcomponents). VAE is excluded per Q3.
5. NO emission sites are wired yet in this sortie — only the storage and propagation plumbing.
6. Verify all types still declare `@unchecked Sendable` (the lock is the entire reason we keep the annotation).

**Context budget**: ~22 turns (R=7 files, C=0, M=7, B=1, L≈250 plumbing, V=5 greps + build). Right-sized but tight; sortie MUST NOT add emission sites.

**Exit criteria**:
- [ ] `make build` succeeds.
- [ ] `make test` succeeds with no new tests added.
- [ ] `grep -R "OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>" Sources/Flux2Core/` returns one match per type listed in Tasks 1–4 (expected: **6 matches** — pipeline + 5 subcomponents; VAE excluded per Q3).
- [ ] `grep -R "func setTelemetry" Sources/Flux2Core/` returns at least 6 matches.
- [ ] `grep -n "setTelemetry" Sources/Flux2Core/VAE/AutoencoderKL.swift` returns empty (VAE has no setter; Q3).
- [ ] `Flux2Pipeline.setTelemetry` body calls `setTelemetry` on every owned subcomponent that has been instantiated at the time of the call (verify by code reading, then by running existing pipeline init test and confirming no crash with `setTelemetry(nil)` followed by `setTelemetry(NoopFlux2TelemetryReporter())`).
- [ ] No production call site invokes `currentTelemetry()` yet (it is `fileprivate` and unused — emit sites land in later sorties). Verify: `grep -R "currentTelemetry()" Sources/Flux2Core/` returns only the 6 declaration lines, no caller sites.
- [ ] `grep -R "^actor " Sources/Flux2Core/Pipeline/ Sources/Flux2Core/Loading/ Sources/Flux2Core/Scheduler/ Sources/Flux2Core/Transformer/` shows no new `actor` declarations (we are explicitly NOT migrating to actors).

---

### Sortie 3: Wire weight-load, LoRA, pipeline-init, and error emissions

**Priority**: 6.0 — dependency depth 5, foundation 0, risk 2 (multiple call sites; errors must precede every throw).

**Agent assignment**: sub-agent (NO builds). Supervising agent runs `make build` after Group C converges.

**Entry criteria**:
- [ ] Sortie 2 exit criteria all met (lock seam exists everywhere).

**Tasks**:
1. In `Sources/Flux2Core/Loading/WeightLoader.swift`, implement a private `dtypeHistogram(_ params:) -> [String: Int]` builder that iterates loaded params and buckets by dtype string (e.g. `"float16"`, `"float32"`, `"int8"`, `"int4"`).
2. Around `loadTextEncoder` (~`Flux2Pipeline.swift:182`), `loadTransformer` (~`:261`), and `loadVAE` (~`:432`), and the internal `Flux2WeightLoader` load calls: emit `weightLoadStart(component:, path:)` before the load, `weightLoadComplete(component:, paramCount:, dtypeHistogram:, sizeMB:, durationSeconds:)` after.
3. Around `loadLoRA(_ config:)` (~`Flux2Pipeline.swift:357`): emit `loraLoadStart` / `loraLoadComplete`.
4. In the `generate*` exit path where deferred LoRA unmerge runs: emit `loraUnmerged(restoredLayerCount:)`.
5. In `Flux2Pipeline.init` (~`:150` end): emit `pipelineInit(model:, quantization:, vaeConfig:, memoryOptimization:)`. Add a new `public func dispose() async` method (Q4 resolved — `deinit` cannot be `async`, so we expose explicit tear-down) that emits `pipelineDispose` and clears `transformer`/`vae`/`textEncoder`/`kleinEncoder`. Do NOT emit `pipelineDispose` from `deinit`. Document in the doc-comment that hosts (Vinetas) should call `dispose()` before releasing the pipeline.
6. Every `throw Flux2Error.…` site in `Flux2Pipeline.swift` (actual count: **14** at audit time, per `grep -c "throw Flux2Error" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`; the REQUIREMENTS spec's enumerated list is non-exhaustive): emit `errorThrown(phase:, errorDescription:, stepIndex:)` IMMEDIATELY before the throw.
7. Emission template MUST be: `if let telemetry = currentTelemetry() { let stat = TuberiaTensorStat.sample(…); await telemetry.capture(.…(…)) }`. NO bare `await reporter.capture(...)` outside an `if let` guard.

**Context budget**: ~28 turns (R=2 large files, C=0, M=3, B=0 — sub-agent, L≈400, V=several greps). Right-sized.

**Exit criteria** (verified by supervising agent after Group C converges):
- [ ] `make build` succeeds (run by supervising agent post-merge).
- [ ] `make test` succeeds (existing suite still green).
- [ ] `grep -nR "telemetry.capture(.weightLoadStart" Sources/Flux2Core/` returns at least 3 sites (text encoder, transformer, VAE).
- [ ] `grep -nR "telemetry.capture(.weightLoadComplete" Sources/Flux2Core/` returns the same count.
- [ ] `grep -c "throw Flux2Error" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` equals `grep -c "telemetry.capture(.errorThrown" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` (audit-time value: 14; if the source has drifted, both must move together).
- [ ] `grep -nR "telemetry.capture(.pipelineInit" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly one match.
- [ ] `grep -nR "telemetry.capture(.pipelineDispose" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly one match, and that match is inside the body of `public func dispose() async` (NOT inside `deinit`). `grep -nR "deinit" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns no telemetry references.
- [ ] `grep -n "public func dispose() async" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly one match.

---

### Sortie 4: Wire text-encoder, VLM, and scheduler emissions

**Priority**: 5.5 — dependency depth 5, foundation 0, risk 2 (multiple encoder variants with subtle naming).

**Agent assignment**: sub-agent (NO builds).

**Entry criteria**:
- [ ] Sortie 2 exit criteria all met. (Sortie 4 is parallel-eligible with Sorties 3 and 5; all three depend only on the lock seam.)

**Tasks**:
1. Around `textEncoder!.encodeWithPrompt(...)`, `textEncoder!.encode(...)`, and `kleinEncoder!.encode(...)` call sites in `generateWithResult` and the I2I branches: emit `textEncoderForwardStart(encoderName:, promptLength:, upsampleRequested:)` before, `textEncoderForwardComplete(encoderName:, finalPromptLength:, embeddingStat:, durationSeconds:)` after. **`encoderName` mapping (Q5 resolved — encoder-family granularity)**: `Flux2TextEncoder` (Mistral) → `"mistral"`, `KleinTextEncoder` (Qwen3) → `"qwen3"`, `TrainingTextEncoder` → `"qwen3-training"`. Match the `WeightComponent` enum naming so phase suffixes line up across event types. Model variant is already carried on `pipelineInit`; no need to repeat it here.
2. Around `textEncoder!.describeImagePathsForPrompt(...)` and `upsamplePromptWithImages(...)`: emit `vlmInterpretStart(imageCount:, encoderUsed:)` and `vlmInterpretComplete(descriptionsProduced:, totalDescriptionLength:, durationSeconds:)`.
3. Inside `FlowMatchEulerScheduler.setTimesteps(...)`, AFTER it computes `mu` via `computeEmpiricalMu(imageSeqLen:numSteps:)` and populates `sigmas`: emit `schedulerConfigured(numTrainTimesteps:, numInferenceSteps:, shift:, imageSeqLen:, mu:, sigmasHead: Array(sigmas.prefix(5)), sigmasTail: Array(sigmas.suffix(5)))` exactly once per call.
4. Use the same `if let telemetry = currentTelemetry()` template; sample `embeddingStat` only inside the guard.

**Context budget**: ~22 turns. Right-sized.

**Exit criteria** (verified by supervising agent after Group C converges):
- [ ] `make build` succeeds (post-merge).
- [ ] `make test` succeeds.
- [ ] `grep -nR "telemetry.capture(.textEncoderForwardStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift | wc -l` ≥ 1.
- [ ] `grep -nR "telemetry.capture(.schedulerConfigured" Sources/Flux2Core/Scheduler/FlowMatchEulerScheduler.swift` returns exactly one match.
- [ ] For every `TuberiaTensorStat.sample(` call this sortie introduces, the immediately preceding line within 3 lines above contains `if let telemetry` (record exceptions in PR description if any).

---

### Sortie 5: Wire VAE-decode, anomaly, and cancellation emissions

**Priority**: 6.5 — dependency depth 5, foundation 0, risk 2.5 (the VAE BatchNorm-denormalize site is "the single most load-bearing math step for image quality" per spec).

**Agent assignment**: sub-agent (NO builds).

**Entry criteria**:
- [ ] Sortie 2 exit criteria all met. (Parallel-eligible with Sorties 3 and 4.)

**Tasks**:
1. Before VAE forward in the decode path (~`Flux2Pipeline.swift:1252`): emit `vaeDecodeStart(latentStat:, scalingFactor:)`.
2. **BatchNorm denormalize emission (CLARIFIED)**: the spec anchors `vaeBatchNormDenormalize` at the `CRITICAL: Denormalize patchified latents with VAE BatchNorm AFTER denoising` comment (currently at `Flux2Pipeline.swift:1422`). The actual call to `LatentUtils.denormalizeLatentsWithBatchNorm` lives at **5 call sites**: `:1146`, `:1226`, `:1385` (mid-loop user-facing CHECKPOINTS — these emit NOTHING), and `:1258`, `:1425` (FINAL-decode denormalize for I2I and T2I paths respectively — these emit). Sample `beforeStat` immediately before the call at 1258 / 1425, run the existing util call, sample `afterStat` immediately after, then emit `vaeBatchNormDenormalize(beforeStat:, afterStat:)`. **Exactly one of {1258, 1425} fires per generation, depending on path.**
3. After `postprocessVAEOutput` succeeds: emit `vaeDecodeComplete(pixelStat:, outputDims:, durationSeconds:)`.
4. **Implement `numericalAnomaly` side-channel (Q6 resolved — helper lives in flux, NOT in SwiftTuberia)**: create `Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` exporting `enum Flux2AnomalyDetector { static func anomalies(in stat: TuberiaTensorStat) -> [Flux2TelemetryEvent.AnomalyKind] }` that returns `[.nan]` when `stat.hasNaN`, `[.inf]` when `stat.hasInf`, `[.outOfRange]` when `abs(stat.max) > TuberiaTensorStat.defaultOutOfRangeThreshold || abs(stat.min) > defaultOutOfRangeThreshold`, `[.zeroLatent]` when `abs(stat.mean) < 1e-6 && stat.std < 1e-6` (latents/embeddings only — gated by caller), `[.dtypeUnexpected]` when caller passes an expected dtype that mismatches. After every emission that carries a stat, the call site does `for kind in Flux2AnomalyDetector.anomalies(in: stat) { await telemetry.capture(.numericalAnomaly(phase: "<source event>", kind: kind, stepIndex:, stat:)) }`. This stays in flux to avoid an SwiftTuberia API expansion just for one consumer's heuristic.
5. At every cancellation check site (REQUIREMENTS calls out `~Flux2Pipeline.swift:1071`): emit `generationCancelled(stepIndex:)`.
6. All samples produced for this sortie's events are inside `if let telemetry` guards.

**Context budget**: ~26 turns. Right-sized.

**Exit criteria** (verified by supervising agent after Group C converges):
- [ ] `make build` succeeds (post-merge).
- [ ] `make test` succeeds.
- [ ] `grep -nR "telemetry.capture(.vaeBatchNormDenormalize" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly **two** matches — one per final-decode path (1258 / 1425), not the three checkpoint sites (1146 / 1226 / 1385). Runtime invariant: **exactly one fires per generation**.
- [ ] `grep -nR "telemetry.capture(.vaeDecodeStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least one match.
- [ ] `grep -nR "telemetry.capture(.numericalAnomaly" Sources/Flux2Core/` returns at least one match.
- [ ] `Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` exists and contains the `anomalies(in:)` helper.
- [ ] `grep -nR "telemetry.capture(.generationCancelled" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns at least one match.

---

### Sortie 6: Wire the HOT-PATH `denoiseStepComplete` and loop-boundary emissions

> **Riskiest emission sortie in the plan.** This is the inner loop of the densest math in Vinetas. The agent MUST read only the four denoise-loop bodies in `Flux2Pipeline.swift` — NOT the full 1875-line file. Atomicity is critical.

**Priority**: 8.5 — dependency depth 4, foundation 0, risk 3 (hot path; the entire +1% overhead target depends on this sortie's discipline).

**Agent assignment**: supervising agent ONLY (must run AFTER Group C; same file as Sortie 3, different functions). Sub-agents can't take this — too risky for drift.

**Entry criteria**:
- [ ] Sortie 2 exit criteria all met (lock seam exists).
- [ ] Sorties 3, 4, 5 are COMPLETED (so the agent does not see drift from concurrent edits to `Flux2Pipeline.swift`).
- [ ] `make build` is green on the merge of Group C.

**Pre-read targeting (context discipline)**:
- Read ONLY these line ranges in `Flux2Pipeline.swift`:
  - **KVExtract one-shot site** (`imageToImageKVExtractStep0`): around `transformer.forwardKVExtract` at `:1079` (read ±30 lines).
  - **KV-cached loop** (`imageToImageKVCached`): around `for stepIdx in 1..<…` at `:1105` (read ±40 lines).
  - **Full-recompute loop** (`imageToImageFullRecompute`): around `for stepIdx in 0..<…` at `:1169` (read ±40 lines).
  - **T2I loop** (`textToImage`): around `for stepIdx in 0..<…` at `:1327` (read ±40 lines).
- Do NOT read other functions. Do NOT read `Transformer/` files. The agent has no business in the transformer block math.

**The 3-loops + 1-one-shot reality (corrects spec's "4 loops" framing)**:
Audit at refinement time found only **3** `for stepIdx in` loops (lines 1105, 1169, 1327). The `imageToImageKVExtractStep0` variant is a **single non-loop call** to `transformer.forwardKVExtract(...)` at `:1079`. Per Q-extract resolution: emit a single triplet (`denoiseLoopStart` + `denoiseStepComplete(stepIndex: 0, totalSteps: 1)` + `denoiseLoopEnd`) around that one call. The KV-cached loop then continues the run from `stepIdx = 1`.

**Tasks**:
1. Identify the three `for stepIdx in` loops by grepping (expected exactly 3 matches: 1105, 1169, 1327).
2. For each of the three loops (variants: `imageToImageKVCached`, `imageToImageFullRecompute`, `textToImage`):
   - IMMEDIATELY before the `for stepIdx in …` line, emit `denoiseLoopStart(variant:, totalSteps:, latentShape:, latentDtype:, initialLatentStat:)`. Sample `initialLatentStat` inside the `if let telemetry` guard. For `imageToImageKVCached`, `totalSteps` is the loop range count (sigma count minus 1, excluding the extract step which is its own triplet — see Task 3).
   - At the TOP of the loop body, add `let telemetry = currentTelemetry()` — exactly one lock acquisition per step. All stat samples within the body use this cached optional.
   - At the bottom of each step body (after `noisePred` is computed AND `scheduler.step` has run, so `latentsBefore` and `latentsAfter` are both available): if `let telemetry`, sample `latentBeforeStat`, `noisePredStat`, `latentAfterStat`, then `await telemetry.capture(.denoiseStepComplete(variant:, stepIndex:, totalSteps:, sigma:, timestep:, latentBeforeStat:, noisePredStat:, latentAfterStat:, kvCacheLayerCount:, kvCacheHit:, durationSeconds:))`.
   - IMMEDIATELY after the loop (success or break-on-cancel): emit `denoiseLoopEnd(variant:, totalSteps:, completedSteps:, finalLatentStat:, durationSeconds:)`.
3. **`imageToImageKVExtractStep0` one-shot at `:1079`**: wrap the `transformer.forwardKVExtract(...)` call in a triplet — emit `denoiseLoopStart(variant: .imageToImageKVExtractStep0, totalSteps: 1, latentShape:, latentDtype:, initialLatentStat:)` immediately before the call, then `denoiseStepComplete(variant: .imageToImageKVExtractStep0, stepIndex: 0, totalSteps: 1, sigma:, timestep:, latentBeforeStat:, noisePredStat: <stat of noisePred0 returned by forwardKVExtract>, latentAfterStat: <stat of the post-extract latent>, kvCacheLayerCount: <kvCache.layerCount>, kvCacheHit: nil, durationSeconds:)` immediately after, then `denoiseLoopEnd(variant: .imageToImageKVExtractStep0, totalSteps: 1, completedSteps: 1, finalLatentStat:, durationSeconds:)`. All three emissions in this one-shot are inside the same `if let telemetry` guard.
4. **`kvCacheLayerCount` / `kvCacheHit` policy (Q7 resolved)**:
   - `textToImage`: both `nil`.
   - `imageToImageKVExtractStep0`: `kvCacheLayerCount = kvCache.layerCount` (read from the `TransformerKVCache` returned by `forwardKVExtract`), `kvCacheHit = nil`.
   - `imageToImageKVCached`: `kvCacheLayerCount = kvCache.layerCount`; `kvCacheHit = true` by default. Emit `kvCacheHit = false` only if a queried block returns `nil` from both `kvCache.doubleStreamEntry(at:)` and `kvCache.singleStreamEntry(at:)` for any block index that was previously populated in this generation. (The optional return type from `TransformerKVCache` API confirms misses are detectable; if instrumenting the lookup site is out of scope for this sortie, hardcode `true` and file the false-detection refinement as a follow-up.)
   - `imageToImageFullRecompute`: both `nil`.
5. `durationSeconds` per step is measured from a `Date()` taken at the top of the loop body to one taken just before `telemetry.capture`. The capture itself is OUTSIDE the timing window. See Q2 for the `@autoclosure` vs. explicit-guard rationale.
6. **DO NOT** sample any tensor stat outside the `if let telemetry` guard. **DO NOT** call `currentTelemetry()` more than once per step.

**Context budget**: ~32 turns (R=4 line-range reads of the same file, C=0, M=1 file in 4 places, B=1, L≈200 added, V=many greps). Right-sized given the pre-read targeting above. Without pre-read targeting it would be oversized at 50+ turns.

**Exit criteria**:
- [ ] `make build` succeeds.
- [ ] `make test` succeeds (existing suite green; new tests land in Sorties 7a/7b/8/9).
- [ ] `grep -nR "telemetry.capture(.denoiseStepComplete" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift | wc -l` returns exactly **4** (3 in-loop step emissions + 1 one-shot for KVExtractStep0).
- [ ] `grep -nR "telemetry.capture(.denoiseLoopStart" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift | wc -l` returns exactly **4**.
- [ ] `grep -nR "telemetry.capture(.denoiseLoopEnd" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift | wc -l` returns exactly **4**.
- [ ] `grep -nR "for stepIdx in" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift | wc -l` returns exactly **3** (1105, 1169, 1327 — sanity: we have not added or lost any loop).
- [ ] `grep -n "transformer.forwardKVExtract" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` returns exactly one match, and the KVExtractStep0 triplet is wrapped around it.
- [ ] Every `TuberiaTensorStat.sample` call within `Flux2Pipeline.swift` is lexically inside an `if let telemetry` block — verify by reading the diff and recording line numbers in the PR description. (No automated grep proves this; the agent must produce the diff line-by-line.)
- [ ] In each of the 3 denoise loop bodies, `currentTelemetry()` appears exactly once. In the KVExtractStep0 one-shot, `currentTelemetry()` also appears exactly once (shared across the triplet).

---

### Sortie 7a: Weight-load and denoise-step functional tests

**Priority**: 4.5 — depth 1, foundation 1 (introduces the shared `MockReporter` used by 7b/8), risk 1.

**Agent assignment**: sub-agent (NO builds).

**Entry criteria**:
- [ ] Sortie 6 is COMPLETED. All emission sites are live.

**Tasks**:
1. **`MockReporter` location (Q8 resolved — `Tests/TestHelpers/` already exists)**: add the helper to `Tests/TestHelpers/MockTelemetryReporter.swift` next to the existing `MockFlux2Pipeline.swift`, so `Flux2CoreTests` and `Flux2GPUTests` can share it (both targets already depend on `TestHelpers` per `Package.swift:91-108`). Implement as an `actor` or `@unchecked Sendable` class with a lock that records every captured event into an array readable from tests.
2. **Weight-fixture-free histogram test (Q8 resolved — no GB-scale downloads in CI)**: add `Tests/Flux2CoreTests/Flux2TelemetryWeightLoadHistogramTests.swift`. Do NOT load real Klein4B weights in this test. Instead, exercise the `dtypeHistogram` builder added in Sortie 3 against a hand-constructed `[String: MLXArray]` of small in-memory tensors with known dtypes. Real-fixture coverage moves to `Tests/Flux2GPUTests/` if/when it's added later (out of scope for this sortie).
3. Add `Tests/Flux2CoreTests/Flux2TelemetryDenoiseStepTests.swift` — run 4 steps of T2I through `MockReporter` using mocked transformer + scheduler (no real weights). Assert: 1× `denoiseLoopStart`, 4× `denoiseStepComplete`, 1× `denoiseLoopEnd`; `stepIndex` is monotone 0..3; `latentBeforeStat` at step N+1 equals `latentAfterStat` at step N.

**Context budget**: ~22 turns. Right-sized.

**Exit criteria** (verified by supervising agent post-Group-E):
- [ ] `make test` succeeds with `Flux2TelemetryWeightLoadHistogramTests` and `Flux2TelemetryDenoiseStepTests` passing.
- [ ] `grep -nR "class MockTelemetryReporter\|actor MockTelemetryReporter" Tests/TestHelpers/` returns exactly one declaration.
- [ ] Both test files exist and are picked up by the test target. Neither test downloads or loads multi-GB model fixtures (Q8: CI-safe).

---

### Sortie 7b: KV-cache, anomaly, and VAE denormalization functional tests

**Priority**: 4.0 — depth 1, foundation 0, risk 1.5 (anomaly injection requires a transformer mock).

**Agent assignment**: sub-agent (NO builds).

**Entry criteria**:
- [ ] Sortie 6 is COMPLETED.
- [ ] `MockTelemetryReporter` exists at `Tests/TestHelpers/MockTelemetryReporter.swift` from Sortie 7a (Q8 alternative: split prerequisite if Sortie 7b dispatches before 7a finishes — then 7b waits on 7a's `MockTelemetryReporter` artifact specifically).

**Tasks**:
1. Add `Tests/Flux2CoreTests/Flux2TelemetryKVCacheHitTests.swift` — Klein9BKV run; assert step 0 fires `denoiseStepComplete(kvCacheHit: nil)` (extraction step), steps 1+ fire `kvCacheHit: true`.
2. Add `Tests/Flux2CoreTests/Flux2TelemetryAnomalyTests.swift` — inject a transformer mock that returns a tensor with one NaN at step 2. Assert: `denoiseStepComplete` at step 2 carries `noisePredStat.hasNaN == true`; `numericalAnomaly(phase: "denoiseStepComplete", kind: .nan, stepIndex: 2, …)` fires alongside.
3. Add `Tests/Flux2CoreTests/Flux2TelemetryVAEDenormalizationTests.swift` — assert `vaeBatchNormDenormalize` fires exactly once per generation and `afterStat.std != beforeStat.std`.

**Context budget**: ~24 turns. Right-sized.

**Exit criteria** (verified by supervising agent post-Group-E):
- [ ] `make test` succeeds with all three new test files passing.
- [ ] All three test files exist and are picked up by the test target.

---

### Sortie 8: Lock-contention TSan test

**Priority**: 5.0 — depth 1, foundation 0, risk 2.5 (the safety net for the entire `@unchecked Sendable` design choice).

**Agent assignment**: sub-agent (NO builds).

**Entry criteria**:
- [ ] Sortie 2 (lock seam) is COMPLETED. (Sortie 8 is parallel-eligible with Sorties 7a/7b — it does not need the emission sites wired.)

**Tasks**:
1. Add `Tests/Flux2CoreTests/Flux2TelemetryLockContentionTests.swift`.
2. Spin up 10 concurrent tasks that call `setTelemetry(...)` with toggling reporters (`nil`, `NoopFlux2TelemetryReporter()`, a `MockTelemetryReporter`).
3. Concurrently run a short mocked denoise loop that internally calls `currentTelemetry()`.
4. Assert no data races (the test must be runnable under TSan and pass; the test itself just asserts emissions reflect the most-recently-set reporter at the time of the read).
5. **Add `make test-tsan` target (Q9 resolved — Makefile has no TSan target today)**: this sortie is the one allowed to edit the Makefile. Add a target that runs `xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -enableThreadSanitizer YES -only-testing Flux2CoreTests/Flux2TelemetryLockContentionTests` with the canonical `ARCHS=arm64 ONLY_ACTIVE_ARCH=YES` flags from `TESTING_REQUIREMENTS.md`. Document the target in the test-file header comment: "Run under TSan via `make test-tsan`."

**Context budget**: ~16 turns. Right-sized.

**Exit criteria** (verified by supervising agent post-Group-E):
- [ ] `make test` runs `Flux2TelemetryLockContentionTests` and it passes.
- [ ] `make test-tsan` exists as a Makefile target and runs `Flux2TelemetryLockContentionTests` under TSan, passing.
- [ ] The test file exists at the expected path.
- [ ] The PR description (produced in Sortie 10) includes the result of one `make test-tsan` run for this test.

---

### Sortie 9: Baseline overhead test (must run on ARM64 hardware, runs alone)

> **This sortie is isolated because it is timing-sensitive.** A 20-iteration wall-clock comparison cannot share an agent with other unrelated edits; any concurrent work can perturb the measurement.

**Priority**: 7.0 — depth 1, foundation 0, risk 3 (the +1% / +5% bound is the public contract of the entire instrumentation; a regression here invalidates the design).

**Agent assignment**: supervising agent ONLY, ARM64 hardware required.

**Entry criteria**:
- [ ] Sortie 6 (hot-path emissions) is COMPLETED.
- [ ] Agent has access to ARM64 Apple Silicon hardware (MLX/Metal). If running this sortie on x86 / CI without ARM, defer per the deferred-sortie rule in `skill.md`.

**Tasks**:
1. Add `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift`.
2. **Mocked-transformer rig (Q10 resolved — no real-weights run)**: a real-weights 20-step T2I × 60 iterations would take minutes per cohort and bloat CI. Instead, build a constant-time transformer stub conforming to whatever interface `Flux2Pipeline` consumes (or invoke the denoise-loop body in isolation) that returns a deterministic, pre-allocated `MLXArray` per step. The overhead test then measures **telemetry overhead per step**, not MLX kernel time — which is what the +1% / +5% bounds are actually about. Single fixed prompt, fixed seed, 20 steps per iteration.
3. Run 20 iterations with `setTelemetry(nil)` and 20 iterations with `setTelemetry(NoopFlux2TelemetryReporter())`. Take wall-clock median of each cohort.
4. Assert the Noop cohort's median is within **+1%** of the nil cohort's median.
5. Run a third cohort of 20 iterations with `setTelemetry(MockTelemetryReporter())`. Assert the Mock cohort's median is within **+5%** of the nil cohort's median.
6. Capture the three medians and their ratios as machine-readable output (e.g. a single stdout line `OVERHEAD_NOOP_PCT=… OVERHEAD_MOCK_PCT=…`). The release PR description (Sortie 10) will paste these.

**Context budget**: ~20 turns. Right-sized; isolation prevents perturbation, not context bloat.

**Exit criteria**:
- [ ] `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift` exists.
- [ ] The test passes on ARM64 hardware with the +1% / +5% bounds.
- [ ] Two clean back-to-back runs produce `OVERHEAD_NOOP_PCT` values within 0.5 percentage points of each other.
- [ ] Numeric results captured for inclusion in the PR description.

---

### Sortie 10: Release — open PR, tag minor version, publish for Vinetas pin

**Priority**: 8.0 — depth 0, foundation 0 (downstream-facing), risk 2 (release-gate work).

**Agent assignment**: supervising agent only.

**Entry criteria**:
- [ ] Sorties 7a, 7b, 8, 9 are all COMPLETED.
- [ ] `make build` and `make test` are green on `instrumentation/01`.

**Tasks**:
1. Push `instrumentation/01` and open a PR against the `main` branch of `intrusive-memory/flux-2-swift-mlx` (Q11 resolved — `git remote -v` confirmed `https://github.com/intrusive-memory/flux-2-swift-mlx.git`).
2. PR description MUST include:
   - Link back to REQUIREMENTS-instrumentation.md.
   - The captured `OVERHEAD_NOOP_PCT` / `OVERHEAD_MOCK_PCT` results from Sortie 9.
   - The `make test-tsan` result from Sortie 8.
   - An explicit note that **SwiftTuberia ≥ 0.7.0** is required (Q1 resolved — see Sortie 1's `Package.swift` edit).
3. Wait for CI green; address review feedback if any (in a new sortie if substantive).
4. Merge to `main`.
5. Bump the package to our next minor release version (additive change per REQUIREMENTS §9). Tag the resulting commit. Current is `3.1.1-dev` per `defe4fe` — first additive-instrumentation release is **`v3.2.0`** unless the user overrides.
6. Create a GitHub release. Note in the release body: "SwiftVinetas should pin flux-2-swift-mlx ≥ v3.2.0."

**Exit criteria**:
- [ ] PR is merged to the default branch.
- [ ] A new minor-version git tag exists on the merge commit (verify with `git tag --list --sort=-v:refname | head -1`).
- [ ] A GitHub release referencing the new tag is published.
- [ ] The PR description contains overhead numbers and a SwiftVinetas-pin note.

---

## Resolved Questions (was: Open Questions & Missing Documentation)

All 11 questions are resolved as of `questions_resolved: 2026-05-12`. No outstanding human-input items remain before dispatch. Decisions are baked into the sortie task / exit-criteria text above; this table is the audit log.

| # | Was-blocking | Decision | Where it lives in the plan |
|---|--------------|----------|----------------------------|
| **Q1** | Sortie 1 | **SwiftTuberia `v0.7.0` is shipped** (verified `git ls-remote --tags` on 2026-05-12; type `TuberiaTensorStat` is in `Sources/Tuberia/Telemetry/TuberiaTensorStat.swift` with the same `if let telemetry` discipline this plan assumes). Sortie 1 adds `sibling("SwiftTuberia", remote: …, from: "0.7.0")` to `Package.swift` mirroring the existing `SwiftAcervo` pattern, and adds the `Tuberia` product to the `Flux2Core` target. | Sortie 1 Tasks #1, Exit criterion #2; Repo constraints. |
| **Q2** | Sorties 3–6 | **Every `TuberiaTensorStat.sample` call MUST be lexically inside `if let telemetry = currentTelemetry() { … }`** — explicit guard, no `@autoclosure`. SwiftTuberia v0.7.0's own docstring confirms this discipline ("`sample(_:)` performs no telemetry side-effects of its own — the only way to get zero-cost-when-off semantics is to keep `sample(_:)` out of the hot path entirely when telemetry is `nil`"). | Sortie 6 Exit criterion (diff inspection); reiterated in every emission sortie. |
| **Q3** | Sortie 2 | **VAE (`AutoencoderKLFlux2`) gets NO setter.** All VAE events (`vaeDecodeStart`/`vaeBatchNormDenormalize`/`vaeDecodeComplete`) fire from inside `Flux2Pipeline` around VAE calls, never from inside the VAE class. Sortie 2's setter list = 6 types total. | Sortie 2 Tasks #2 & #4, Exit criteria. |
| **Q4** | Sortie 3 | **Explicit `public func dispose() async` method on `Flux2Pipeline`** emits `pipelineDispose`; `deinit` emits nothing. Hosts (Vinetas) must call `dispose()` before releasing the pipeline. | Sortie 3 Task #5, Exit criteria. |
| **Q5** | Sortie 4 | **Encoder-family naming**: Mistral → `"mistral"`, Qwen3 (Klein) → `"qwen3"`, training → `"qwen3-training"`. Matches `WeightComponent` enum naming so phase suffixes line up across event types. Model variant already carried on `pipelineInit`. | Sortie 4 Task #1. |
| **Q6** | Sortie 5 | **Anomaly detector helper lives in flux** — `Sources/Flux2Core/Telemetry/Flux2AnomalyDetector.swift` exposes `static func anomalies(in stat: TuberiaTensorStat) -> [AnomalyKind]`. Each emission site that carries a stat loops over the result. Avoids needing to expand SwiftTuberia's API. | Sortie 5 Task #4, Exit criterion. |
| **Q7** | Sortie 6 | **`kvCacheHit` policy pinned**: `nil` for textToImage / imageToImageFullRecompute / KVExtractStep0; `true` by default for imageToImageKVCached; `false` only when both `kvCache.doubleStreamEntry(at:)` and `kvCache.singleStreamEntry(at:)` return `nil` for a previously-populated block index. The `TransformerKVCache` API supports detection. | Sortie 6 Task #4. |
| **Q8** | Sortie 7a | **Helper lives at `Tests/TestHelpers/MockTelemetryReporter.swift`** (next to existing `MockFlux2Pipeline.swift`). Histogram & step tests do NOT load real Klein weights — they exercise builders against hand-constructed in-memory tensors. Real-fixture coverage is `Flux2GPUTests` scope, deferred. | Sortie 7a Tasks #1–#3. |
| **Q9** | Sortie 8 | **Sortie 8 adds `make test-tsan`** to the Makefile (the one Makefile edit allowed in this campaign). Target invokes `xcodebuild test … -enableThreadSanitizer YES -only-testing Flux2CoreTests/Flux2TelemetryLockContentionTests`. | Sortie 8 Task #5. |
| **Q10** | Sortie 9 | **Mocked-transformer overhead rig** — a real-weights 60-iter Klein4B run would bloat CI to minutes; the test should measure telemetry overhead per step, not MLX kernel time. Use a constant-time transformer stub returning pre-allocated deterministic `MLXArray`s. | Sortie 9 Task #2. |
| **Q11** | Sortie 10 | **Repo remote**: `https://github.com/intrusive-memory/flux-2-swift-mlx.git` (default branch `main`). Confirmed via `git remote -v`. | Sortie 10 Task #1. |

## Audit-time discrepancies fixed in this revision

While resolving Q-blockers, three plan-vs-reality mismatches were corrected:

1. **Denoise loops: 3, not 4.** `grep -c "for stepIdx in" Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` = 3 (lines 1105, 1169, 1327). The `imageToImageKVExtractStep0` variant is a single non-loop call to `transformer.forwardKVExtract(...)` at `:1079`. Sortie 6 now emits a triplet around that one call (denoiseLoopStart + denoiseStepComplete(stepIndex:0, totalSteps:1) + denoiseLoopEnd). Exit-criteria counts updated: 4 stepComplete + 4 loopStart + 4 loopEnd = `3 in-loop + 1 one-shot`.
2. **BatchNorm denormalize: 5 call sites, only 2 emit.** `grep -n denormalizeLatentsWithBatchNorm` = 5 sites (1146, 1226, 1258, 1385, 1425). Sites 1146/1226/1385 are mid-loop checkpoint emits and do NOT carry a `vaeBatchNormDenormalize` event. Sites 1258 (I2I final-decode) and 1425 (T2I final-decode) emit; exactly one fires per generation depending on path. The spec's "line 1422" reference is the comment immediately above 1425.
3. **`throw Flux2Error` count: 14, not 9.** Sortie 3 exit criterion now requires `grep -c throw Flux2Error == grep -c telemetry.capture(.errorThrown` regardless of which absolute value the audit returned.

**Verdict**: No outstanding blockers. Plan is ready for `mission-supervisor start`.

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 7 |
| Total sorties | 10 |
| Dependency structure | layered (1 → 2 → {3 ∥ 4 ∥ 5} → 6 → {7a ∥ 7b ∥ 8 ∥ 9} → 10) |
| Critical path length | 5 sorties (1 → 2 → 6 → 7a → 10) |
| Parallel-eligible sets | {3, 4, 5} after Sortie 2; {7a, 7b, 8} after Sortie 6; {9} runs alone for timing fidelity |
| Agent allocation | 1 supervising agent + up to 3 sub-agents |
| Sub-agent build restriction | Sorties 3, 4, 5, 7a, 7b, 8 are sub-agent eligible and NEVER run builds; supervising agent runs `make build` / `make test` between groups |
| Hot-path sortie | 6 (denoiseStepComplete) — isolated by design |
| Lock seam sortie | 2 — isolated by design (highest single-piece risk per host plan) |
| Timing-sensitive sortie | 9 — runs alone on ARM64 hardware |
| Open questions blocking dispatch | 0 (Q1–Q11 all resolved 2026-05-12) |
| Open questions resolvable as first sortie task | 0 |

## Refinement Pass Results

| Pass | Status | Changes |
|------|--------|---------|
| 1. Atomicity & Testability | PASS | Sortie 6 explicitly bounded by line-range pre-read targeting (was at risk of oversize on a 1875-line file). Sortie 7 split into 7a/7b to keep each sortie's test count ≤3 with one shared artifact (`MockReporter`). All exit criteria are machine-verifiable greps, file-exists checks, or build/test commands. |
| 2. Prioritization | PASS | Sortie 2 (lock seam) and Sortie 6 (hot-path) carry the highest risk scores (3.0) and reflect the campaign's two biggest single-point-of-failure risks. Layer order is unchanged from breakdown — dependencies forbid earlier movement of any sortie. |
| 3. Parallelism | PASS | 7 work groups, with parallel groups {3, 4, 5} and {7a, 7b, 8}. Sub-agent count capped at 3 concurrent (max budget is 4 — leaving 1 slot of headroom). Build-restriction policy: only the supervising agent runs `make build` / `make test`; sub-agents touch only source files. Critical path = 5 sorties. |
| 4. Open Questions | PASS WITH FLAGS | 11 open questions identified. Q1, Q2, Q3, Q4 are dispatch-blockers (must resolve before their sortie's entry criteria are met). Q2 (`@autoclosure` discipline) is the single most load-bearing question — its resolution is baked into every emission sortie's exit criteria. Q5–Q11 are first-task questions the sortie agent resolves before producing code. |
| 5. Questions Resolved | PASS (2026-05-12) | All 11 questions resolved (see Resolved Questions section). SwiftTuberia v0.7.0 confirmed published with `TuberiaTensorStat`. Decision summary: Q1 add `from: "0.7.0"` sibling; Q2 explicit guard everywhere; Q3 no VAE setter; Q4 explicit `dispose() async`; Q5 encoder-family naming; Q6 anomaly helper in flux; Q7 default-true with detectable miss; Q8 Tests/TestHelpers + mocked loader; Q9 add `make test-tsan`; Q10 mocked transformer; Q11 confirmed `intrusive-memory/flux-2-swift-mlx`. Three audit-time discrepancies (loop count, BatchNorm sites, throw count) corrected. |

**VERDICT**: Plan is refined, all questions resolved, and ready for execution. No human input required before `mission-supervisor start`.
