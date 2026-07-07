// Int4DirectLoadGPUTests.swift — Klein 4B int4 load routing (GPU)
//
// HISTORY: This suite originally validated the R2.3 "no phys_footprint spike"
// acceptance for the DIRECT pre-quantized int4 transformer load
// (`themindstudio/flux2-klein-4b-mlx-4bit`, mflux 0.15.2) — packed MLX 4-bit
// weights loaded STRAIGHT into QuantizedLinear with no bf16 intermediate.
//
// That direct-load path empirically produced PURE NOISE. `(klein4B, .int4)` now
// resolves to `.klein4B_bf16` and takes the proven bf16 + on-the-fly `quantize()`
// path (the same route Klein 9B uses): load the full-precision bf16 transformer
// via the mature diffusers→Swift remapper, then quantize to 4-bit in-process.
// The memory optimization of the old direct load is intentionally traded away
// for a correct image, so the "< 8 GB no-spike" ceiling no longer applies.
//
// This suite now proves the NEW routing (bf16, not pre-quantized) and that a
// small int4 generation completes and returns an image of the requested size.
// It REQUIRES the bf16 Klein 4B transformer, the VAE, and a Qwen3 encoder on
// disk, so it is MODEL-PRESENCE-GATED via `.enabled(if:)` and SKIPS cleanly
// when the weights are absent (running in CI against cached SwiftAcervo models).

import CoreGraphics
import Foundation
import Metal
import SwiftAcervo
import TestHelpers
import Testing

@testable import Flux2Core

/// True only when a Metal device is present AND all three components needed to
/// drive the int4 (bf16 + on-the-fly quantize) transformer load — the bf16
/// Klein 4B transformer, the VAE, and a Qwen3 text encoder — are on disk.
func int4DirectLoadTestEnabled() -> Bool {
  guard MTLCreateSystemDefaultDevice() != nil else { return false }

  // Crash-guard: `Acervo.isModelAvailable` fatalErrors when neither
  // ACERVO_MODELS_DIR nor an App Group is configured (unentitled local runner).
  let env = ProcessInfo.processInfo.environment
  let hasModelsDir = (env[Acervo.modelsDirectoryOverrideVariable]?.isEmpty == false)
  let hasAppGroup = (env[Acervo.appGroupEnvironmentVariable]?.isEmpty == false)
  guard hasModelsDir || hasAppGroup else { return false }

  // Derive the transformer repoId from the registry so a rename can't silently
  // desync this gate: (klein4B, .int4) → .klein4B_bf16 → black-forest-labs/FLUX.2-klein-4B.
  let transformerRepoId =
    ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4).repoId
  let vaeRepoId = ModelRegistry.VAEVariant.standard.repoId
  let qwen3_8bit = "lmstudio-community/Qwen3-4B-MLX-8bit"
  let qwen3_4bit = "lmstudio-community/Qwen3-4B-MLX-4bit"

  return Acervo.isModelAvailable(transformerRepoId)
    && Acervo.isModelAvailable(vaeRepoId)
    && (Acervo.isModelAvailable(qwen3_8bit) || Acervo.isModelAvailable(qwen3_4bit))
}

@Suite("Klein 4B int4 → bf16 + on-the-fly quantize (GPU)", .serialized)
struct Int4DirectLoadGPUTests {

  /// (klein4B, .int4) resolves to bf16 (on-the-fly quantize), not the
  /// noise-producing pre-quantized mflux 4-bit variant, and a small int4
  /// generation completes and returns an image of the requested size.
  @Test(
    "Klein 4B int4 loads bf16 + on-the-fly quantize and generates an image",
    .enabled(if: int4DirectLoadTestEnabled()),
    .timeLimit(.minutes(10))
  )
  func int4LoadsBf16AndGenerates() async throws {
    // ultraMinimal = { textEncoder: .mlx4bit, transformer: .int4 } — the int4
    // path. Prove the variant resolution + routing before spending inference
    // time (belt-and-suspenders with the pure-logic core tests).
    let quantization = Flux2QuantizationConfig.ultraMinimal
    #expect(quantization.transformer == .int4)
    let variant = ModelRegistry.TransformerVariant.variant(
      for: .klein4B, quantization: quantization.transformer)
    #expect(
      variant == .klein4B_bf16,
      "(klein4B, .int4) must resolve to bf16 (on-the-fly quantize), not the noise-producing direct load"
    )
    #expect(
      !variant.isPreQuantizedMLX,
      "int4 path must take the bf16 + on-the-fly quantize branch, not the pre-quantized direct load"
    )

    // Capture telemetry so we can assert the on-the-fly quantization actually
    // ran (a quantizationComplete event for the transformer).
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: quantization)
    pipeline.setTelemetry(reporter)

    try await pipeline.loadModels()

    // The transformer is loaded lazily during generation; a tiny generation
    // triggers loadTransformer() → bf16 load → on-the-fly quantize(bits: 4).
    let width = 512
    let height = 512
    let image = try await pipeline.generateTextToImage(
      prompt: "a single red apple on a white table",
      height: height,
      width: width,
      steps: 4,
      guidance: 1.0,
      seed: 7
    )

    // Generation returned a real image of the requested dimensions.
    #expect(image.width == width)
    #expect(image.height == height)

    // The on-the-fly quantization ran (transformer took the bf16 + quantize path).
    let events = await reporter.snapshot()
    var sawTransformerQuantization = false
    for event in events {
      if case .quantizationComplete(let component, let bits, _, _) = event,
        component == .transformer
      {
        sawTransformerQuantization = true
        #expect(bits == 4, "int4 path must quantize the transformer to 4-bit on-the-fly")
      }
    }
    #expect(
      sawTransformerQuantization,
      "expected a transformer quantizationComplete event from the on-the-fly quantize path")
  }
}
