@preconcurrency import MLX
import Foundation
import Tuberia  // for TuberiaTensorStat

/// Telemetry event surface emitted from inside `Flux2Pipeline` (and its owned
/// subcomponents) so a host adapter (Vinetas) can correlate every numerical
/// anomaly back to the kernel and step that produced it.
///
/// Re-uses `TuberiaTensorStat` from `Tuberia` — this enum does NOT define its
/// own stat variant. See `REQUIREMENTS-instrumentation.md` §3.1.
public enum Flux2TelemetryEvent: Sendable {

    // MARK: - Pipeline lifecycle

    /// Emitted from `Flux2Pipeline.init`.
    ///
    /// F7: `init` is sync; `capture` is async. The emission site dispatches
    /// via `Task { ... }`, so this event may not fire if the host has not
    /// called `setTelemetry()` yet. Hosts should call `setTelemetry()` before
    /// the first generation to receive this event.
    case pipelineInit(model: String, quantization: QuantizationManifest, vaeConfig: String, memoryOptimization: String)

    /// Emitted from `Flux2Pipeline.dispose()` (a new async method).
    /// `deinit` cannot be async, so explicit tear-down via `dispose()` is
    /// required for this event to fire.
    case pipelineDispose(model: String)

    // MARK: - Model / weight loading (boundary memory events)

    case weightLoadStart(component: WeightComponent, path: String)
    case weightLoadComplete(component: WeightComponent, paramCount: Int, dtypeHistogram: [String: Int], sizeMB: Double, durationSeconds: Double)

    // MARK: - LoRA

    case loraLoadStart(name: String, scale: Double)
    case loraLoadComplete(name: String, adapterParamCount: Int, mergedLayerCount: Int, sizeMB: Double, durationSeconds: Double)
    case loraUnmerged(restoredLayerCount: Int)

    // MARK: - Text encoding (per encoder type)

    case textEncoderForwardStart(encoderName: String, promptLength: Int, upsampleRequested: Bool)
    case textEncoderForwardComplete(encoderName: String, finalPromptLength: Int, embeddingStat: TuberiaTensorStat, durationSeconds: Double)

    // MARK: - VLM interpretation (when interpretImagePaths is non-nil)

    case vlmInterpretStart(imageCount: Int, encoderUsed: String)
    case vlmInterpretComplete(descriptionsProduced: Int, totalDescriptionLength: Int, durationSeconds: Double)

    // MARK: - Scheduler

    case schedulerConfigured(numTrainTimesteps: Int, numInferenceSteps: Int, shift: Float, imageSeqLen: Int, mu: Float, sigmasHead: [Float], sigmasTail: [Float])

    // MARK: - Denoise loop (memory-boundary events on start / end)

    case denoiseLoopStart(variant: DenoiseVariant, totalSteps: Int, latentShape: [Int], latentDtype: String, initialLatentStat: TuberiaTensorStat)
    case denoiseLoopEnd(variant: DenoiseVariant, totalSteps: Int, completedSteps: Int, finalLatentStat: TuberiaTensorStat, durationSeconds: Double)

    // MARK: - Per-step denoise (the high-frequency event)

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

    // MARK: - VAE decode (memory boundary on complete)

    case vaeDecodeStart(latentStat: TuberiaTensorStat, scalingFactor: Float)
    /// The CRITICAL "Denormalize patchified latents with VAE BatchNorm AFTER
    /// denoising" step at `Flux2Pipeline.swift:~1422`. See REQUIREMENTS §5
    /// (F6): emit only at the 2 final-decode sites, NOT at the 3 mid-loop
    /// checkpoint sites.
    case vaeBatchNormDenormalize(beforeStat: TuberiaTensorStat, afterStat: TuberiaTensorStat)
    case vaeDecodeComplete(pixelStat: TuberiaTensorStat, outputDims: [Int], durationSeconds: Double)

    // MARK: - Numerical anomaly side-channel

    case numericalAnomaly(phase: String, kind: AnomalyKind, stepIndex: Int?, stat: TuberiaTensorStat)

    // MARK: - Cancellation

    /// F3: `stepIndex` is `Int?`. Pre-loop cancellation sites use `nil`;
    /// in-loop sites use the current `stepIdx`. Do NOT use sentinel values
    /// like `0` or `-1` for pre-loop sites.
    case generationCancelled(stepIndex: Int?)

    // MARK: - Error side-channel

    case errorThrown(phase: ErrorPhase, errorDescription: String, stepIndex: Int?)

    // MARK: - Nested types

    public struct QuantizationManifest: Sendable, Codable {
        public let textEncoder: String      // e.g. "q4-g64"
        public let transformer: String      // e.g. "q4-g64" / "fp16" / "fp8"
        public let vae: String              // typically "fp16" or "fp32"
        public init(textEncoder: String, transformer: String, vae: String) {
            self.textEncoder = textEncoder
            self.transformer = transformer
            self.vae = vae
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
        case textToImage                     // Loop at ~Flux2Pipeline.swift:1327
        case imageToImageKVExtractStep0      // Single non-loop call at ~Flux2Pipeline.swift:1079 (transformer.forwardKVExtract); emits ONE denoiseStepComplete triplet with stepIndex:0 totalSteps:1
        case imageToImageKVCached            // Loop at ~Flux2Pipeline.swift:1105 (steps 1..N after extract)
        case imageToImageFullRecompute       // Loop at ~Flux2Pipeline.swift:1169
    }

    public enum AnomalyKind: String, Sendable {
        case nan
        case inf
        case outOfRange      // default threshold: TuberiaTensorStat.defaultOutOfRangeThreshold
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
        case imageProcessingFailed  // F2: added for Flux2Error.imageProcessingFailed
        case other
    }
}
