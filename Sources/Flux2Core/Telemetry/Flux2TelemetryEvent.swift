@preconcurrency import MLX
import Foundation
import Tuberia  // for TuberiaTensorStat

public enum Flux2TelemetryEvent: Sendable {

    // --- Pipeline lifecycle ---
    // pipelineInit: emitted from `Flux2Pipeline.init`.
    // pipelineDispose: emitted from a new `public func dispose() async` method on `Flux2Pipeline`,
    // NOT from `deinit` (deinit cannot be async). Hosts (Vinetas) must call `dispose()` before
    // releasing the pipeline if they want a tear-down event.
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
        case textToImage                     // Loop at ~Flux2Pipeline.swift:1327
        case imageToImageKVExtractStep0      // Single non-loop call at ~Flux2Pipeline.swift:1079 (transformer.forwardKVExtract); emits ONE denoiseStepComplete triplet with stepIndex:0 totalSteps:1
        case imageToImageKVCached            // Loop at ~Flux2Pipeline.swift:1105 (steps 1..N after extract)
        case imageToImageFullRecompute       // Loop at ~Flux2Pipeline.swift:1169
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
