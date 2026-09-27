import Flux2Core
import FluxTextEncoders
import Foundation
import Metal
import SwiftAcervo
// GPUPreconditions.swift — Shared GPU guards for Flux2GPUTests
import Testing

func checkGPUPreconditions(minimumBytes: UInt64) -> Bool {
  guard MTLCreateSystemDefaultDevice() != nil else {
    Issue.record("No Metal device available")
    return false
  }
  guard ProcessInfo.processInfo.physicalMemory >= minimumBytes else {
    Issue.record(
      "Insufficient memory: \(ProcessInfo.processInfo.physicalMemory) bytes, need \(minimumBytes)")
    return false
  }
  return true
}

// MARK: - Shared serialized parent

/// Parent for the GPU suites that load Klein 4B through the process-wide
/// `FluxTextEncoders.shared` singleton (Flux2CoreGPUTests,
/// FluxTextEncodersGPUTests). `.serialized` is inherited by the nested suites,
/// so no two of these tests load/unload the shared encoder or a multi-GB
/// pipeline concurrently.
@Suite(.serialized) enum KleinModelGPUTests {}

// MARK: - `.enabled(if:)` gates (skip, never fail, when the host can't run a test)

/// Metal device present and at least `minimumBytes` of physical memory.
func hostHasGPU(minimumBytes: UInt64) -> Bool {
  MTLCreateSystemDefaultDevice() != nil
    && ProcessInfo.processInfo.physicalMemory >= minimumBytes
}

/// `Acervo` resolves its models directory from `ACERVO_MODELS_DIR` or an App
/// Group identifier and `fatalError`s when neither is configured (the
/// unentitled local test-runner case). Check before calling any `Acervo` API.
func acervoIsConfigured() -> Bool {
  let env = ProcessInfo.processInfo.environment
  return env[Acervo.modelsDirectoryOverrideVariable]?.isEmpty == false
    || env[Acervo.appGroupEnvironmentVariable]?.isEmpty == false
}

/// The Klein 4B Qwen3 text encoder on disk (8-bit preferred, 4-bit fallback,
/// matching `KleinTextEncoder.findDownloadedQwen3Variant`), or `nil`.
func downloadedKlein4BQwen3Variant() -> Qwen3Variant? {
  guard acervoIsConfigured() else { return nil }
  return [Qwen3Variant.qwen3_4B_8bit, .qwen3_4B_4bit].first {
    Acervo.isModelAvailable($0.repoId)
  }
}

/// True when a ≥16 GB Metal host has every Klein 4B pipeline component
/// (transformer, VAE, Qwen3 text encoder) on disk. Replaces the old
/// `KLEIN_MODEL_PATH` env-var guard, which nothing set, so those tests either
/// silently passed without running or recorded a failure.
func klein4BPipelineTestsEnabled() -> Bool {
  guard hostHasGPU(minimumBytes: 16 * 1_073_741_824), acervoIsConfigured() else { return false }
  let transformerRepoId =
    ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4).repoId
  return Acervo.isModelAvailable(transformerRepoId)
    && Acervo.isModelAvailable(ModelRegistry.VAEVariant.standard.repoId)
    && downloadedKlein4BQwen3Variant() != nil
}

/// True when a ≥16 GB Metal host has a Klein 4B Qwen3 text encoder on disk.
func klein4BTextEncoderTestsEnabled() -> Bool {
  hostHasGPU(minimumBytes: 16 * 1_073_741_824) && downloadedKlein4BQwen3Variant() != nil
}
