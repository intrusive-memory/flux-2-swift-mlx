# flux-2-swift-mlx — Instrumentation Requirements

**Status:** Draft, awaiting implementation
**Pattern source:** [Vinetas `docs/INSTRUMENTATION_PLAN.md`](https://github.com/intrusive-memory/Vinetas/blob/development/docs/INSTRUMENTATION_PLAN.md) + Produciesta `Docs/TELEMETRY_IMPL_PATTERN.md`
**Host:** Vinetas
**Depends on:** SwiftTuberia ≥ 0.7.0 (for the shared `TuberiaTensorStat` type — flux re-uses it instead of defining its own)
**Priority:** **P0** — densest math surface in the Vinetas dep graph; the library that actually produces NaN/Inf when something goes wrong

---

## 1. Why instrument flux-2-swift-mlx

`Flux2Pipeline` is where the diffusion math actually runs: text encoder forward, FlowMatchEuler scheduler init + per-step ODE update, transformer denoise step (the 1875-line workhorse class), and VAE decode. **Every numerical anomaly visible to Vinetas — gray images, oversaturated outputs, NaN cascades, dtype-mismatch artifacts — originates here.** SwiftTuberia's per-step events tell you *that* something went wrong; flux-2-swift-mlx's events tell you *which kernel produced it*.

The instrumentation must surface:

- **Quantization configuration at load.** Klein4B/Klein9B/Klein9BKV/Klein4BBase/Dev each have different transformer + text-encoder + VAE quant levels. Bug reports about "wrong colors" most often correlate with an unexpected quant combination.
- **Text encoder forward outcomes.** Klein uses Qwen3 via `KleinTextEncoder`; Dev uses Mistral via `Flux2TextEncoder`. They produce embeddings with different shapes, dtypes, and value distributions. The embedding `TensorStat` at the boundary is the first signal that's actually a tensor.
- **FlowMatchEuler internal state.** `computeEmpiricalMu(imageSeqLen:, numSteps:)` is a non-obvious math step at scheduler init — the `mu` value affects every sigma in the schedule. Snapshotting it is cheap and high-value.
- **Per-step denoise math.** For each of the four denoise variants (T2I, I2I with KV-extract step 0, I2I with KV-cached steps 1..N, I2I full-recompute), emit one event per step with: `step`, `sigma`, `latentBefore`, `noisePred`, `latentAfter`, optionally `kvCacheLayerCount`.
- **KV-cache lifecycle.** Klein9BKV ships with KV-cache; if cache misses on step N when it should hit, only telemetry will tell us.
- **VAE decode boundary.** The VAE is famous for producing out-of-range pixels when fed a bad latent. The `latentStat` going in and `pixelStat` coming out is the single most diagnostic image-fault correlation.
- **The CRITICAL `Denormalize patchified latents with VAE BatchNorm AFTER denoising` step** (existing comment at line 1422). Wrong order here makes images look "muddy." Telemetry must record that this happened.
- **LoRA merge state.** Whether base or merged weights are active for this generation.

What it must NOT surface:
- Per-transformer-block events. The 24+ blocks would explode event count without diagnostic value at this layer (anomalies show up in the per-step `noisePred` stat regardless of which block produced them).
- Per-attention-head events. Same reason.
- VLM internals (the image-interpretation flow). VLM is a separate code path that runs *before* generation; its output is captured implicitly via `finalUsedPrompt`.
- The `Flux2Profiler` timings (already a separate observability seam — keep as-is).

---

## 2. Coexistence with existing surfaces

| Surface | Purpose | Status |
|---|---|---|
| `Flux2Debug.log` / `Flux2Debug.verbose` | Free-form developer log output to stdout | **Keep as-is.** Telemetry events fire alongside; no consolidation. |
| `Flux2Profiler.start/end(label)` | Timing buckets ("2. Text Encoding", "7. VAE Decode") | **Keep as-is.** Telemetry events carry their own `durationSeconds` measured from `Date()` deltas inside the emission guard. Profiler remains the human-readable summary. |
| `Flux2DownloadProgressCallback` (model download progress) | UI progress | **Keep as-is.** Adapter on the Vinetas side can correlate against the `weightLoadComplete` event. |
| Step-level progress callback inside `generate*` methods | UI per-step progress | **Keep as-is.** Telemetry events fire from the same callsites but to a different sink. |
| `Flux2Error` `LocalizedError` cases | User-facing errors | **Keep as-is.** Each `throw Flux2Error.…` site also fires `errorThrown` immediately before the throw. |

---

## 3. Public types to add

```
Sources/Flux2Core/Telemetry/
  Flux2TelemetryEvent.swift
  Flux2TelemetryReporter.swift
```

`TuberiaTensorStat` is imported from SwiftTuberia (`import Tuberia`) — flux does not define its own variant. The dep already exists.

### 3.1 `Flux2TelemetryEvent.swift`

```swift
@preconcurrency import MLX
import Foundation
import Tuberia  // for TuberiaTensorStat

public enum Flux2TelemetryEvent: Sendable {

    // --- Pipeline lifecycle ---
    case pipelineInit(model: String, quantization: QuantizationManifest, vaeConfig: String, memoryOptimization: String)
    case pipelineDispose

    // --- Model / weight loading (boundary memory events) ---
    case weightLoadStart(component: WeightComponent, path: String)
    case weightLoadComplete(component: WeightComponent, paramCount: Int, dtypeHistogram: [String: Int], sizeMB: Double, durationSeconds: Double)
    // Adapter routes ALL weightLoadComplete events through captureWithMemorySnapshot.

    // --- LoRA ---
    case loraLoadStart(name: String, scale: Double)
    case loraLoadComplete(name: String, adapterParamCount: Int, mergedLayerCount: Int, sizeMB: Double, durationSeconds: Double)
    case loraUnmerged(restoredLayerCount: Int)

    // --- Text encoding (per encoder type) ---
    case textEncoderForwardStart(encoderName: String, promptLength: Int, upsampleRequested: Bool)
    case textEncoderForwardComplete(encoderName: String, finalPromptLength: Int, embeddingStat: TuberiaTensorStat, durationSeconds: Double)

    // --- VLM interpretation (when interpretImagePaths is non-nil) ---
    case vlmInterpretStart(imageCount: Int, encoderUsed: String)
    case vlmInterpretComplete(descriptionsProduced: Int, totalDescriptionLength: Int, durationSeconds: Double)

    // --- Scheduler ---
    case schedulerConfigured(numTrainTimesteps: Int, numInferenceSteps: Int, shift: Float, imageSeqLen: Int, mu: Float, sigmasHead: [Float], sigmasTail: [Float])

    // --- Denoise loop (memory-boundary events on start / end) ---
    case denoiseLoopStart(variant: DenoiseVariant, totalSteps: Int, latentShape: [Int], latentDtype: String, initialLatentStat: TuberiaTensorStat)
    case denoiseLoopEnd(variant: DenoiseVariant, totalSteps: Int, completedSteps: Int, finalLatentStat: TuberiaTensorStat, durationSeconds: Double)
    // Adapter routes denoiseLoopStart and denoiseLoopEnd through captureWithMemorySnapshot.

    // --- Per-step denoise (the high-frequency event) ---
    case denoiseStepComplete(
        variant: DenoiseVariant,
        stepIndex: Int,
        totalSteps: Int,
        sigma: Float,
        timestep: Float,
        latentBeforeStat: TuberiaTensorStat,
        noisePredStat: TuberiaTensorStat,
        latentAfterStat: TuberiaTensorStat,
        kvCacheLayerCount: Int?,
        kvCacheHit: Bool?,
        durationSeconds: Double
    )

    // --- VAE decode (memory boundary on complete) ---
    case vaeDecodeStart(latentStat: TuberiaTensorStat, scalingFactor: Float)
    case vaeBatchNormDenormalize(beforeStat: TuberiaTensorStat, afterStat: TuberiaTensorStat)  // see Flux2Pipeline.swift:1422
    case vaeDecodeComplete(pixelStat: TuberiaTensorStat, outputDims: [Int], durationSeconds: Double)
    // Adapter routes vaeDecodeComplete through captureWithMemorySnapshot.

    // --- Numerical anomaly side-channel ---
    case numericalAnomaly(phase: String, kind: AnomalyKind, stepIndex: Int?, stat: TuberiaTensorStat)

    // --- Cancellation ---
    case generationCancelled(stepIndex: Int)

    // --- Error side-channel ---
    case errorThrown(phase: ErrorPhase, errorDescription: String, stepIndex: Int?)

    public struct QuantizationManifest: Sendable, Codable {
        public let textEncoder: String      // e.g. "q4-g64"
        public let transformer: String      // e.g. "q4-g64" / "fp16" / "fp8"
        public let vae: String              // typically "fp16" or "fp32"
        public init(textEncoder: String, transformer: String, vae: String) {
            self.textEncoder = textEncoder; self.transformer = transformer; self.vae = vae
        }
    }

    public enum WeightComponent: String, Sendable {
        case textEncoderKlein     // Qwen3 (KleinTextEncoder)
        case textEncoderDev       // Mistral (DevTextEncoder / Flux2TextEncoder)
        case textEncoderTraining  // TrainingTextEncoder
        case transformer
        case vae
        case lora
    }

    public enum DenoiseVariant: String, Sendable {
        case textToImage
        case imageToImageKVExtractStep0      // ~Flux2Pipeline.swift:1079
        case imageToImageKVCached            // ~Flux2Pipeline.swift:1105–1167
        case imageToImageFullRecompute       // ~Flux2Pipeline.swift:1169–1241
    }

    public enum AnomalyKind: String, Sendable {
        case nan
        case inf
        case outOfRange      // default threshold: |x| > 1e6
        case zeroLatent      // mean ≈ 0 && std ≈ 0 (model didn't produce anything)
        case dtypeUnexpected // dtype != configured dtype
    }

    public enum ErrorPhase: String, Sendable {
        case modelNotLoaded
        case invalidConfiguration
        case insufficientMemory
        case modelNotDownloaded
        case generationCancelled
        case generationFailed
        case weightLoadFailed
        case vaeDecodeFailed
        case textEncoderFailed
        case vlmInterpretFailed
        case loraLoadFailed
        case other
    }
}
```

**`dtypeHistogram` rationale.** A transformer load that produces `{"float16": 800, "float32": 12, "int8": 0}` for Klein4B's transformer is correct. `{"float16": 0, "float32": 812}` is the smoking gun for "weights silently dequantized." One small histogram per component, six-ish entries — cheaper than a full dump and answers the right question.

### 3.2 `Flux2TelemetryReporter.swift`

```swift
public protocol Flux2TelemetryReporter: Sendable {
    func capture(_ event: Flux2TelemetryEvent) async
}

public struct NoopFlux2TelemetryReporter: Flux2TelemetryReporter {
    public init() {}
    public func capture(_ event: Flux2TelemetryEvent) async {}
}
```

---

## 4. Injection points

### 4.1 The `@unchecked Sendable` problem

`Flux2Pipeline` is `public class Flux2Pipeline: @unchecked Sendable` (`Flux2Pipeline.swift:72`). Same for `KleinTextEncoder` (`KleinTextEncoder.swift:22`), `DevTextEncoder` (`DevTextEncoder.swift:15`), `FlowMatchEulerScheduler` (`FlowMatchEulerScheduler.swift:34`), and the transformer classes. The reporter cannot be stored as a plain `var` because that breaks `Sendable`.

**Pattern:** store telemetry behind an `OSAllocatedUnfairLock`:

```swift
import os.lock

public class Flux2Pipeline: @unchecked Sendable {
    private let _telemetryLock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(initialState: nil)

    public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?) {
        _telemetryLock.withLock { $0 = reporter }
    }

    fileprivate func currentTelemetry() -> (any Flux2TelemetryReporter)? {
        _telemetryLock.withLock { $0 }
    }
}
```

Emission template (replacing the autoclosure guard for actor types):

```swift
if let telemetry = currentTelemetry() {
    let stat = TuberiaTensorStat.sample(latents)
    await telemetry.capture(.denoiseStepComplete(/* … */))
}
```

The lock is uncontended in the steady state (single writer at run boundary, single reader inside the pipeline). `OSAllocatedUnfairLock.withLock` is a ~10ns operation; the cost is dominated by the actual reporter call afterward.

### 4.2 Setters on each top-level type

| Type | File:Line | Setter |
|---|---|---|
| `Flux2Pipeline` | `Flux2Pipeline.swift:72` | `public func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?)` |
| `KleinTextEncoder` | `KleinTextEncoder.swift:22` | same |
| `DevTextEncoder` | `DevTextEncoder.swift:15` | same |
| `Flux2TextEncoder` (Mistral) | search `Loading/MistralEncoder.swift` | same |
| `FlowMatchEulerScheduler` | `FlowMatchEulerScheduler.swift:34` | same |
| `Flux2WeightLoader` | `WeightLoader.swift:9` | same; weight loader is per-pipeline so its setter is called from `Flux2Pipeline.setTelemetry` for free |
| `Flux2Transformer2DModel` | `Transformer/Flux2Transformer.swift` | same — fires `weightLoadComplete(.transformer)` from inside its load path |

The host (Vinetas adapter) only ever calls `Flux2Pipeline.setTelemetry`. The pipeline is responsible for propagating the reporter to its owned text encoders / scheduler / transformer / VAE.

### 4.3 Why not migrate to an `actor`?

A migration from `@unchecked Sendable class` to `actor` is the right long-term answer, but it would touch every call site and be a breaking API change. The lock-based seam is a non-breaking additive change. Migration can happen separately.

---

## 5. Per-event emission spec

| Event | Emission site (file:line) | Notes |
|---|---|---|
| `pipelineInit` | `Flux2Pipeline.init` end (~`Flux2Pipeline.swift:150`) | One per pipeline construction. |
| `weightLoadStart` / `Complete` | Around `loadTextEncoder` (`:182`), `loadTransformer` (`:261`), `loadVAE` (`:432`), and `Flux2WeightLoader` internal calls | One pair per `WeightComponent`. `dtypeHistogram` built from `MLXArray.dtype` of each loaded parameter. |
| `loraLoadStart` / `Complete` | Around `loadLoRA(_ config:)` (`:357`) | 0–1 pair per generate. |
| `loraUnmerged` | After deferred unmerge in `generate*` exit path | 0–1 per generate. |
| `textEncoderForwardStart` / `Complete` | Around `textEncoder!.encodeWithPrompt(...)`, `textEncoder!.encode(...)`, `kleinEncoder!.encode(...)` — at the call sites in `generateWithResult` and the I2I branches | 1–2 per generate. `encoderName` populated from `model` enum value. |
| `vlmInterpretStart` / `Complete` | Around `textEncoder!.describeImagePathsForPrompt(...)` and `upsamplePromptWithImages(...)` | 0–1 per generate. |
| `schedulerConfigured` | After `scheduler.setTimesteps(...)` returns | Once per generate. Carry `mu` (computed by `computeEmpiricalMu`), sigmasHead = first 5 + sigmasTail = last 5. |
| `denoiseLoopStart` | Just before the `for stepIdx in ...` loop (one site per variant: ~`:1105`, `:1169`, `:1327`) | **Memory snapshot.** Carries `initialLatentStat` and `variant`. |
| `denoiseStepComplete` | After each `noisePred` + scheduler `step` inside the loop body | n per generate (4–50). Carries all four required stats. |
| `denoiseLoopEnd` | Just after the loop exit (success or break-on-cancel) | **Memory snapshot.** |
| `vaeDecodeStart` | Before VAE forward in `decode` path (~`:1252`) | |
| `vaeBatchNormDenormalize` | At the `CRITICAL: Denormalize patchified latents with VAE BatchNorm AFTER denoising` site (`:1422`) | Stats sampled before & after. **This is the single most load-bearing math step for image quality.** |
| `vaeDecodeComplete` | After `postprocessVAEOutput` succeeds | **Memory snapshot.** Carries `pixelStat` (min should be ~0, max ~1 after normalization) and `outputDims`. |
| `numericalAnomaly` | Fires from inside any `TuberiaTensorStat.sample` whose result has `hasNaN || hasInf || max.magnitude > 1e6` or `(mean.magnitude < 1e-6 && std < 1e-6)` for any latent or embedding event | Side-channel; lives next to the source event in the JSONL. |
| `generationCancelled` | At every cancellation check site, currently around `:1071` | Carries the step index at cancellation. |
| `errorThrown` | Every `throw Flux2Error.…` in `Flux2Pipeline.swift` (lines 285, 438, 543, 585, 655, 697, 773, 1071, 1272, and any others) | Fire immediately before throw. |

### Hot-path discipline

The denoise loop is the densest hot path in the whole Vinetas stack. Each step in T2I issues:
- 1 transformer forward (this is the inner loop of the transformer's blocks — flux's responsibility is the *step*, not the *block*)
- 1 scheduler.step
- 1 `denoiseStepComplete` emit (3 TensorStat samples × 8 reductions = 24 MLX reductions)

`OSAllocatedUnfairLock.withLock` to fetch the reporter must happen exactly **once per step** — cache the result in a local var at the start of the loop body so the multiple stat-sampling calls within the step share one telemetry pointer read.

```swift
for stepIdx in 0..<(scheduler.sigmas.count - 1) {
    let telemetry = currentTelemetry()   // ONE lock acquisition per step
    // ... existing math ...
    if let telemetry {
        let latentBefore = TuberiaTensorStat.sample(latentsBefore)
        let noisePredStat = TuberiaTensorStat.sample(noisePred)
        let latentAfter = TuberiaTensorStat.sample(latents)
        await telemetry.capture(.denoiseStepComplete(/* ... */))
    }
}
```

KV-cache variants must populate `kvCacheLayerCount` and `kvCacheHit` accurately — `kvCacheHit: false` on a step that should hit is a smoking gun for a cache-invalidation bug.

---

## 6. Adapter mapping (Vinetas host side)

`Flux2TelemetryAdapter` at `Vinetas/Telemetry/Adapters/Flux2TelemetryAdapter.swift`:

| Event | Sink phase | Memory snapshot? |
|---|---|---|
| `pipelineInit` | `flux_pipeline_init` | no |
| `pipelineDispose` | `flux_pipeline_dispose` | no |
| `weightLoadStart` | `flux_weight_load_start_<component>` | no |
| `weightLoadComplete` | `flux_weight_load_complete_<component>` | **yes** (per INSTRUMENTATION_PLAN §3.1) |
| `loraLoadStart` / `Complete` | `flux_lora_load_start` / `flux_lora_load_complete` | no |
| `loraUnmerged` | `flux_lora_unmerged` | no |
| `textEncoderForwardStart` | `flux_encode_start_<encoderName>` | no |
| `textEncoderForwardComplete` | `flux_encode_complete_<encoderName>` | no |
| `vlmInterpretStart` / `Complete` | `flux_vlm_interpret_start` / `..._complete` | no |
| `schedulerConfigured` | `flux_scheduler_configured` | no |
| `denoiseLoopStart` | `flux_denoise_loop_start` | **yes** |
| `denoiseStepComplete` | `flux_denoise_step_complete` | no (stepIndex populates Snapshot.stepIndex) |
| `denoiseLoopEnd` | `flux_denoise_loop_end` | **yes** |
| `vaeDecodeStart` | `flux_vae_decode_start` | no |
| `vaeBatchNormDenormalize` | `flux_vae_batchnorm_denormalize` | no |
| `vaeDecodeComplete` | `flux_vae_decode_complete` | **yes** |
| `numericalAnomaly` | `flux_anomaly_<kind>` | no |
| `generationCancelled` | `flux_cancelled` | no |
| `errorThrown` | `flux_error_<phase>` | no |

Exhaustive switch in the adapter; no `default:`.

---

## 7. Tests

Add to `Tests/Flux2CoreTests/`:

| Test | Purpose |
|---|---|
| `Flux2TelemetryWeightLoadHistogramTests` | Load Klein4B weights through a `MockReporter`; assert `weightLoadComplete(.transformer)` carries a `dtypeHistogram` whose dominant key is the configured quant level (e.g. `"q4-g64"` → many `int8` and `int4` entries). |
| `Flux2TelemetryDenoiseStepTests` | Run 4 steps of T2I (mocked or real) through `MockReporter`. Assert: 1× `denoiseLoopStart`, 4× `denoiseStepComplete`, 1× `denoiseLoopEnd`; stepIndex monotone; `latentBeforeStat` at step N+1 equals `latentAfterStat` at step N (same tensor). |
| `Flux2TelemetryNoopOverheadTests` | 10-step T2I run with `nil` reporter vs `NoopFlux2TelemetryReporter`. Wall-clock medians within ±2% over 20 iterations. This is the single most important test for proving the hot-path guard works. |
| `Flux2TelemetryKVCacheHitTests` | Klein9BKV run; assert step 0 fires `denoiseStepComplete(kvCacheHit: nil)` (extraction step), steps 1+ fire `kvCacheHit: true`. |
| `Flux2TelemetryAnomalyTests` | Inject a transformer mock that returns a tensor with one NaN at step 2. Assert: `denoiseStepComplete` at step 2 carries `noisePredStat.hasNaN == true`; `numericalAnomaly(phase: "flux_denoise_step_complete", kind: .nan, stepIndex: 2, ...)` fires alongside. |
| `Flux2TelemetryVAEDenormalizationTests` | Assert `vaeBatchNormDenormalize` fires exactly once per generation and `afterStat.std` differs from `beforeStat.std` (proves the math actually ran). |
| `Flux2TelemetryLockContentionTests` | 10 concurrent `setTelemetry` toggles + a running denoise loop; assert no data races (TSan-clean) and step emissions correctly reflect the most-recently-set reporter. |

The lock-contention test is the safety net for the `OSAllocatedUnfairLock`-based seam.

---

## 8. Out of scope

- Per-transformer-block events. ~24+ blocks per step × 4–50 steps = unmanageable.
- Per-attention-head, per-RoPE, per-modulation events.
- `Flux2Profiler` integration. It's a separate observability surface; both stay.
- Training instrumentation (`Sources/Flux2Core/Training/Training.swift`). Vinetas only runs inference. Training telemetry is a future scope.
- Internal `MemoryManager.fullCleanup()` callsites. The boundary memory events already capture the relevant transitions.

---

## 9. Versioning

**Minor** version bump (additive). Pin floor: `3.2.0` post-release. Must ship AFTER SwiftTuberia ≥ 0.7.0 (which publishes `TuberiaTensorStat`).

---

## 10. Implementation checklist

- [ ] Add `Sources/Flux2Core/Telemetry/Flux2TelemetryEvent.swift` per §3.1
- [ ] Add `Sources/Flux2Core/Telemetry/Flux2TelemetryReporter.swift` per §3.2
- [ ] Add `OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>` and `setTelemetry`/`currentTelemetry` to `Flux2Pipeline`, `KleinTextEncoder`, `DevTextEncoder`, `Flux2TextEncoder`, `FlowMatchEulerScheduler`, `Flux2WeightLoader`, transformer classes
- [ ] In `Flux2Pipeline.setTelemetry`, propagate the reporter to all owned subcomponents
- [ ] Wire emission sites per §5; cache `currentTelemetry()` once per step inside the denoise loop
- [ ] Implement `dtypeHistogram` builder inside `Flux2WeightLoader` (iterate loaded params, bucket by dtype string)
- [ ] Ensure every `throw Flux2Error.…` is preceded by `errorThrown` emit
- [ ] Add tests per §7
- [ ] Run baseline overhead test; commit results in PR description (focus: 10-step T2I, nil reporter, ±2% bound)
- [ ] Run TSan on the lock-contention test
- [ ] Tag release with `MINOR` bump
