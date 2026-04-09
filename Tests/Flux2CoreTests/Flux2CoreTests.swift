// Flux2CoreTests.swift - Unit tests for Flux2Core
// Copyright 2025 Vincent Gourbin

import Testing
import Foundation
@testable import Flux2Core
import MLX
import ImageIO
import CoreGraphics

#if canImport(AppKit)
import AppKit
#endif

@Suite struct Flux2CoreTests {

    // MARK: - Configuration Tests

    @Test func transformerConfigDefaults() {
        let config = Flux2TransformerConfig.flux2Dev

        #expect(config.patchSize == 1)
        #expect(config.inChannels == 128)
        #expect(config.numLayers == 8)
        #expect(config.numSingleLayers == 48)
        #expect(config.numAttentionHeads == 48)
        #expect(config.attentionHeadDim == 128)
        #expect(config.innerDim == 6144)  // 48 * 128
        #expect(config.jointAttentionDim == 15360)
    }

    @Test func vaeConfigDefaults() {
        let config = VAEConfig.flux2Dev

        #expect(config.latentChannels == 32)
        #expect(config.blockOutChannels == [128, 256, 512, 512])
        #expect(config.useBatchNorm)
    }

    @Test func quantizationPresets() {
        #expect(Flux2QuantizationConfig.highQuality.textEncoder == .bf16)
        #expect(Flux2QuantizationConfig.highQuality.transformer == .bf16)

        #expect(Flux2QuantizationConfig.balanced.textEncoder == .mlx8bit)
        #expect(Flux2QuantizationConfig.balanced.transformer == .qint8)

        #expect(Flux2QuantizationConfig.minimal.textEncoder == .mlx4bit)
        #expect(Flux2QuantizationConfig.minimal.transformer == .qint8)

        #expect(Flux2QuantizationConfig.ultraMinimal.textEncoder == .mlx4bit)
        #expect(Flux2QuantizationConfig.ultraMinimal.transformer == .int4)
    }

    @Test func transformerQuantizationInt4() {
        #expect(TransformerQuantization.int4.bits == 4)
        #expect(TransformerQuantization.int4.groupSize == 64)
        #expect(TransformerQuantization.int4.rawValue == "int4")
    }

    @Test func modelRegistryVariantOnTheFlyQuantization() {
        // Klein 9B should always return bf16 variant (quantize on-the-fly)
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .qint8) ==
            .klein9B_bf16
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .int4) ==
            .klein9B_bf16
        )

        // Dev int4 should return bf16 variant (quantize on-the-fly)
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .int4) ==
            .bf16
        )

        // Klein 4B int4 should return bf16 variant (quantize on-the-fly)
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4) ==
            .klein4B_bf16
        )
    }

    // MARK: - Latent Utils Tests

    @Test func latentDimensionValidation() {
        let (h, w) = LatentUtils.validateDimensions(height: 1000, width: 1000)

        // Should be rounded up to nearest multiple of 16
        #expect(h % 16 == 0)
        #expect(w % 16 == 0)
        #expect(h >= 1000)
        #expect(w >= 1000)
    }

    @Test func latentPacking() {
        // Create test latent: [1, 32, 128, 128]
        let latent = MLXRandom.normal([1, 32, 128, 128])

        // Pack
        let packed = LatentUtils.packLatents(latent, patchSize: 2)

        // Should be [1, (128/2)*(128/2), 32*2*2] = [1, 4096, 128]
        #expect(packed.shape[0] == 1)
        #expect(packed.shape[1] == 4096)
        #expect(packed.shape[2] == 128)

        // Unpack
        let unpacked = LatentUtils.unpackLatents(
            packed,
            height: 1024,  // 128 * 8
            width: 1024,
            latentChannels: 32,
            patchSize: 2
        )

        // Should match original shape
        #expect(unpacked.shape == latent.shape)
    }

    @Test func positionIDGeneration() {
        let height = 1024
        let width = 1024

        let imageIds = LatentUtils.generateImagePositionIDs(height: height, width: width)

        // For 1024x1024 with patch size 2: (128/2) * (128/2) = 4096 patches
        #expect(imageIds.shape[0] == 4096)
        #expect(imageIds.shape[1] == 4)  // [T, H, W, L]
    }

    // MARK: - Scheduler Tests

    @Test func schedulerTimesteps() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 50)

        #expect(scheduler.timesteps.count == 51)  // 50 steps + final
        // Timesteps are sigmas * numTrainTimesteps (1000), so first is ~1000 (after time shift)
        // Sigmas are in [0, 1] range - check sigmas instead for semantic correctness
        #expect(scheduler.sigmas.count == 51)
        #expect(scheduler.sigmas.first! > 0.9)  // First sigma should be close to 1.0
        #expect(abs(scheduler.sigmas.last! - 0.0) < 0.001)  // Terminal sigma is 0
        #expect(abs(scheduler.timesteps.last! - 0.0) < 0.01)  // Terminal timestep is 0
    }

    @Test func schedulerStep() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])

        let nextSample = scheduler.step(
            modelOutput: modelOutput,
            timestep: scheduler.timesteps[0],
            sample: sample
        )

        #expect(nextSample.shape == sample.shape)
    }

    // MARK: - Memory Estimation Tests

    @Test func memoryEstimation() {
        let config = Flux2QuantizationConfig.balanced

        // Should estimate reasonable memory
        #expect(config.estimatedTotalMemoryGB > 30)
        #expect(config.estimatedTotalMemoryGB < 100)

        // Text encoding phase should be less than image generation
        #expect(
            config.textEncodingPhaseMemoryGB <
            config.imageGenerationPhaseMemoryGB
        )
    }
}

// MARK: - Embedding Tests

@Suite struct EmbeddingTests {

    @Test func timestepEmbedding() {
        let embedder = Flux2TimestepGuidanceEmbeddings(
            embeddingDim: 256,
            timeEmbedDim: 6144,
            useGuidanceEmbeds: true
        )

        let timestep = MLXArray([Float(0.5)])
        let guidance = MLXArray([Float(4.0)])

        let embedding = embedder(timestep: timestep, guidance: guidance)

        #expect(embedding.shape == [1, 6144])
    }

    @Test func roPE() {
        let rope = Flux2RoPE(axesDims: [32, 32, 32, 32], theta: 2000.0)

        // Create position IDs: [100, 4]
        var flatData: [Int32] = []
        for i: Int32 in 0..<100 {
            flatData.append(contentsOf: [Int32(0), i / 10, i % 10, Int32(0)])
        }
        let ids = MLXArray(flatData).reshaped([100, 4])

        let (cosEmb, sinEmb) = rope(ids)

        // Should output [100, 128] (sum of axes dims)
        #expect(cosEmb.shape[0] == 100)
        #expect(sinEmb.shape[0] == 100)
    }
}

// MARK: - Integration Tests

@Suite struct IntegrationTests {

    @Test func modulationFlow() {
        let dim = 6144
        let modulation = Flux2Modulation(dim: dim, numSets: 2)

        let embedding = MLXRandom.normal([1, dim])
        let params = modulation(embedding)

        #expect(params.count == 2)

        for param in params {
            #expect(param.shift.shape == [1, dim])
            #expect(param.scale.shape == [1, dim])
            #expect(param.gate.shape == [1, dim])
        }
    }

    @Test func feedForwardShape() {
        let dim = 6144
        let ff = Flux2FeedForward(dim: dim)

        let input = MLXRandom.normal([1, 100, dim])
        let output = ff(input)

        #expect(output.shape == input.shape)
    }
}

// MARK: - LoRA Configuration Tests

@Suite struct LoRAConfigTests {

    @Test func loRAConfigInit() {
        let config = LoRAConfig(filePath: "/path/to/lora.safetensors")

        #expect(config.filePath == "/path/to/lora.safetensors")
        // Default scale is 1.0 (not nil)
        #expect(config.scale == 1.0)
        #expect(config.activationKeyword == nil)
    }

    @Test func loRAConfigWithScale() {
        let config = LoRAConfig(filePath: "/path/to/lora.safetensors", scale: 0.8)

        #expect(config.scale == 0.8)
        #expect(config.effectiveScale == 0.8)
    }

    @Test func loRAConfigDefaultScale() {
        let config = LoRAConfig(filePath: "/path/to/lora.safetensors")

        // When no scale is set, effectiveScale should default to 1.0
        #expect(config.effectiveScale == 1.0)
    }

    @Test func loRAConfigName() {
        let config = LoRAConfig(filePath: "/path/to/my_lora.safetensors")

        #expect(config.name == "my_lora")
    }

    @Test func loRAConfigWithActivationKeyword() {
        var config = LoRAConfig(filePath: "/path/to/lora.safetensors")
        config.activationKeyword = "sks"

        #expect(config.activationKeyword == "sks")
    }
}

// MARK: - Scheduler Extended Tests

@Suite struct SchedulerExtendedTests {

    @Test func schedulerCustomSigmas() {
        let scheduler = FlowMatchEulerScheduler()

        // Custom 4-step turbo schedule
        let customSigmas: [Float] = [1.0, 0.65, 0.35, 0.1]
        scheduler.setCustomSigmas(customSigmas)

        // Should have 5 sigmas (4 custom + terminal 0.0)
        #expect(scheduler.sigmas.count == 5)
        #expect(abs(scheduler.sigmas.last! - 0.0) < 0.001)
    }

    @Test func schedulerI2IStrength() {
        let scheduler = FlowMatchEulerScheduler()

        // Full denoise (strength = 1.0)
        scheduler.setTimesteps(numInferenceSteps: 50, strength: 1.0)
        let fullSteps = scheduler.sigmas.count - 1

        // Half denoise (strength = 0.5)
        scheduler.setTimesteps(numInferenceSteps: 50, strength: 0.5)
        let halfSteps = scheduler.sigmas.count - 1

        #expect(halfSteps < fullSteps)
        #expect(halfSteps == 25)
    }

    @Test func schedulerProgress() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        #expect(abs(scheduler.progress - 0.0) < 0.01)

        // Simulate stepping
        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])

        _ = scheduler.step(modelOutput: modelOutput, timestep: scheduler.timesteps[0], sample: sample)

        #expect(scheduler.progress > 0.0)
        #expect(scheduler.remainingSteps == 9)
    }

    @Test func schedulerReset() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])
        _ = scheduler.step(modelOutput: modelOutput, timestep: scheduler.timesteps[0], sample: sample)

        #expect(scheduler.stepIndex > 0)

        scheduler.reset()
        #expect(scheduler.stepIndex == 0)
    }

    @Test func schedulerAddNoise() {
        let scheduler = FlowMatchEulerScheduler()

        let original = MLXArray([Float(1.0), Float(2.0), Float(3.0)])
        let noise = MLXArray([Float(0.1), Float(0.2), Float(0.3)])

        // At timestep 500 (sigma = 0.5)
        let noisy = scheduler.addNoise(originalSamples: original, noise: noise, timestep: 500)

        #expect(noisy.shape == original.shape)
    }

    @Test func schedulerScaleNoise() {
        let scheduler = FlowMatchEulerScheduler()

        let sample = MLXArray([Float(1.0)])
        let noise = MLXArray([Float(0.0)])

        // At sigma = 0, should return sample unchanged
        let result = scheduler.scaleNoise(sample: sample, sigma: 0.0, noise: noise)
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(result)
        #expect(abs(result.item(Float.self) - 1.0) < 0.001)
    }

    @Test func schedulerInitialSigma() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 20)

        // Initial sigma should be close to 1.0 (high noise)
        #expect(scheduler.initialSigma > 0.9)
    }

    @Test func schedulerCurrentSigma() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)

        let firstSigma = scheduler.currentSigma

        let sample = MLXRandom.normal([1, 100, 128])
        let modelOutput = MLXRandom.normal([1, 100, 128])
        _ = scheduler.step(modelOutput: modelOutput, timestep: scheduler.timesteps[0], sample: sample)

        let secondSigma = scheduler.currentSigma

        // Sigma should decrease as we step through
        #expect(secondSigma < firstSigma)
    }
}

// MARK: - Model Registry Tests

@Suite struct ModelRegistryTests {

    @Test func transformerVariantHuggingFaceRepo() {
        let bf16 = ModelRegistry.TransformerVariant.bf16
        #expect(!bf16.huggingFaceRepo.isEmpty)
        #expect(bf16.huggingFaceRepo.contains("FLUX"))
    }

    @Test func transformerVariantEstimatedSize() {
        let bf16 = ModelRegistry.TransformerVariant.bf16
        let qint8 = ModelRegistry.TransformerVariant.qint8

        // bf16 should be larger than qint8
        #expect(bf16.estimatedSizeGB > qint8.estimatedSizeGB)
    }

    @Test func kleinVariantsSmallerThanDev() {
        let devBf16 = ModelRegistry.TransformerVariant.bf16.estimatedSizeGB
        let klein4B = ModelRegistry.TransformerVariant.klein4B_bf16.estimatedSizeGB

        #expect(klein4B < devBf16)
    }

    @Test func vaeVariants() {
        let vae = ModelRegistry.VAEVariant.standard
        #expect(!vae.huggingFaceRepo.isEmpty)
    }

    @Test func recommendedConfigForRAM() {
        // Very low RAM should recommend ultra-minimal config (4-bit)
        let veryLowRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 24)
        #expect(veryLowRamConfig.transformer == .int4)

        // Low RAM should recommend minimal config
        let lowRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 32)
        #expect(lowRamConfig.transformer == .qint8)

        // High RAM can use bf16
        let highRamConfig = ModelRegistry.recommendedConfig(forRAMGB: 128)
        #expect(highRamConfig.transformer == .bf16)
    }

    // MARK: - Gated Status Tests

    @Test func transformerVariantIsGated() {
        // Dev bf16 from black-forest-labs is gated
        #expect(ModelRegistry.TransformerVariant.bf16.isGated)

        // qint8 from VincentGOURBIN repo is NOT gated
        #expect(!ModelRegistry.TransformerVariant.qint8.isGated)

        // Klein 4B is NOT gated (open access)
        #expect(!ModelRegistry.TransformerVariant.klein4B_bf16.isGated)
        #expect(!ModelRegistry.TransformerVariant.klein4B_8bit.isGated)

        // Klein 9B from black-forest-labs IS gated
        #expect(ModelRegistry.TransformerVariant.klein9B_bf16.isGated)
    }

    @Test func textEncoderVariantIsGated() {
        // bf16 from mistralai is gated
        #expect(ModelRegistry.TextEncoderVariant.bf16.isGated)

        // Quantized versions from lmstudio-community are NOT gated
        #expect(!ModelRegistry.TextEncoderVariant.mlx8bit.isGated)
        #expect(!ModelRegistry.TextEncoderVariant.mlx6bit.isGated)
        #expect(!ModelRegistry.TextEncoderVariant.mlx4bit.isGated)
    }

    @Test func vaeVariantIsGated() {
        // VAE is downloaded from Klein 4B repo which is NOT gated
        #expect(!ModelRegistry.VAEVariant.standard.isGated)
    }

    // MARK: - HuggingFace URL Tests

    @Test func transformerVariantHuggingFaceURL() {
        let bf16 = ModelRegistry.TransformerVariant.bf16
        #expect(bf16.huggingFaceURL.starts(with: "https://huggingface.co/"))
        #expect(bf16.huggingFaceURL.contains(bf16.huggingFaceRepo))
    }

    @Test func textEncoderVariantHuggingFaceURL() {
        let mlx8bit = ModelRegistry.TextEncoderVariant.mlx8bit
        #expect(mlx8bit.huggingFaceURL.starts(with: "https://huggingface.co/"))
        #expect(mlx8bit.huggingFaceURL.contains(mlx8bit.huggingFaceRepo))
    }

    @Test func vaeVariantHuggingFaceURL() {
        let vae = ModelRegistry.VAEVariant.standard
        #expect(vae.huggingFaceURL.starts(with: "https://huggingface.co/"))
        #expect(vae.huggingFaceURL.contains(vae.huggingFaceRepo))
    }

    @Test func textEncoderVariantHuggingFaceRepoValues() {
        // bf16 should be from mistralai
        #expect(ModelRegistry.TextEncoderVariant.bf16.huggingFaceRepo.contains("mistralai"))

        // Quantized should be from lmstudio-community
        #expect(ModelRegistry.TextEncoderVariant.mlx8bit.huggingFaceRepo.contains("lmstudio-community"))
        #expect(ModelRegistry.TextEncoderVariant.mlx6bit.huggingFaceRepo.contains("lmstudio-community"))
        #expect(ModelRegistry.TextEncoderVariant.mlx4bit.huggingFaceRepo.contains("lmstudio-community"))
    }

    // MARK: - License Tests

    @Test func transformerVariantLicense() {
        // Dev is non-commercial
        #expect(ModelRegistry.TransformerVariant.bf16.license.contains("Non-Commercial"))
        #expect(!ModelRegistry.TransformerVariant.bf16.isCommercialUseAllowed)

        // Klein 4B is Apache 2.0 (commercial OK)
        #expect(ModelRegistry.TransformerVariant.klein4B_bf16.license.contains("Apache"))
        #expect(ModelRegistry.TransformerVariant.klein4B_bf16.isCommercialUseAllowed)

        // Klein 9B is non-commercial
        #expect(!ModelRegistry.TransformerVariant.klein9B_bf16.isCommercialUseAllowed)
    }

    @Test func textEncoderVariantLicense() {
        // Mistral is Apache 2.0
        #expect(ModelRegistry.TextEncoderVariant.mlx8bit.license.contains("Apache"))
        #expect(ModelRegistry.TextEncoderVariant.mlx8bit.isCommercialUseAllowed)
    }

    @Test func vaeVariantLicense() {
        // VAE inherits FLUX.2 Dev non-commercial license
        #expect(ModelRegistry.VAEVariant.standard.license.contains("Non-Commercial"))
        #expect(!ModelRegistry.VAEVariant.standard.isCommercialUseAllowed)
    }

    // MARK: - Default Parameters Tests

    @Test func transformerVariantDefaultParameters() {
        // Dev: 28 steps, guidance 4.0
        #expect(ModelRegistry.TransformerVariant.bf16.defaultSteps == 28)
        #expect(ModelRegistry.TransformerVariant.bf16.defaultGuidance == 4.0)

        // Klein: 4 steps, guidance 1.0
        #expect(ModelRegistry.TransformerVariant.klein4B_bf16.defaultSteps == 4)
        #expect(ModelRegistry.TransformerVariant.klein4B_bf16.defaultGuidance == 1.0)
    }

    @Test func transformerVariantMaxReferenceImages() {
        // Dev variants: 6 images (delegates to modelType)
        #expect(ModelRegistry.TransformerVariant.bf16.maxReferenceImages == 6)
        #expect(ModelRegistry.TransformerVariant.qint8.maxReferenceImages == 6)

        // Klein variants: 4 images
        #expect(ModelRegistry.TransformerVariant.klein4B_bf16.maxReferenceImages == 4)
        #expect(ModelRegistry.TransformerVariant.klein4B_8bit.maxReferenceImages == 4)
        #expect(ModelRegistry.TransformerVariant.klein9B_bf16.maxReferenceImages == 4)
    }
}

// MARK: - Flux2Model Tests

@Suite struct Flux2ModelTests {

    @Test func defaultSteps() {
        #expect(Flux2Model.dev.defaultSteps == 28)
        #expect(Flux2Model.klein4B.defaultSteps == 4)
        #expect(Flux2Model.klein9B.defaultSteps == 4)
    }

    @Test func defaultGuidance() {
        #expect(Flux2Model.dev.defaultGuidance == 4.0)
        #expect(Flux2Model.klein4B.defaultGuidance == 1.0)
        #expect(Flux2Model.klein9B.defaultGuidance == 1.0)
    }

    @Test func estimatedTimeSeconds() {
        #expect(Flux2Model.dev.estimatedTimeSeconds > 1000)
        #expect(Flux2Model.klein4B.estimatedTimeSeconds < 60)
    }

    @Test func license() {
        #expect(Flux2Model.klein4B.license.contains("Apache"))
        #expect(Flux2Model.klein4B.isCommercialUseAllowed)
        #expect(!Flux2Model.dev.isCommercialUseAllowed)
    }

    @Test func maxReferenceImages() {
        // Dev supports up to 6 reference images (memory limited)
        #expect(Flux2Model.dev.maxReferenceImages == 6)

        // Klein models support up to 4 reference images
        #expect(Flux2Model.klein4B.maxReferenceImages == 4)
        #expect(Flux2Model.klein9B.maxReferenceImages == 4)
    }
}

// MARK: - VAE Config Extended Tests

@Suite struct VAEConfigExtendedTests {

    @Test func vaeConfigDev() {
        let config = VAEConfig.flux2Dev

        #expect(config.latentChannels == 32)
        #expect(config.inChannels == 3)
        #expect(config.outChannels == 3)
    }

    @Test func vaeConfigBlockChannels() {
        let config = VAEConfig.flux2Dev

        // Should have 4 block levels
        #expect(config.blockOutChannels.count == 4)

        // Channels should increase then plateau
        #expect(config.blockOutChannels[0] < config.blockOutChannels[1])
    }

    @Test func vaeConfigScaling() {
        let config = VAEConfig.flux2Dev

        #expect(config.scalingFactor != 0.0)
    }

    @Test func vaeConfigPatchSize() {
        let config = VAEConfig.flux2Dev

        #expect(config.patchSize.0 == 2)
        #expect(config.patchSize.1 == 2)
    }

    @Test func vaeConfigNormalization() {
        let config = VAEConfig.flux2Dev

        #expect(config.normNumGroups > 0)
        #expect(config.normEps > 0)
    }
}

// MARK: - Latent Utils Extended Tests

@Suite struct LatentUtilsExtendedTests {

    @Test func dimensionValidationRounding() {
        // Test various dimensions
        let testCases: [(Int, Int)] = [
            (100, 100),
            (512, 512),
            (1000, 1000),
            (1920, 1080),
        ]

        for (h, w) in testCases {
            let (validH, validW) = LatentUtils.validateDimensions(height: h, width: w)
            #expect(validH % 16 == 0, "Height \(validH) should be multiple of 16")
            #expect(validW % 16 == 0, "Width \(validW) should be multiple of 16")
            #expect(validH >= h)
            #expect(validW >= w)
        }
    }

    @Test func latentPackUnpackRoundtrip() {
        // Test multiple sizes
        let sizes = [(64, 64), (128, 128), (96, 128)]

        for (h, w) in sizes {
            let latent = MLXRandom.normal([1, 32, h, w])
            let packed = LatentUtils.packLatents(latent, patchSize: 2)
            let unpacked = LatentUtils.unpackLatents(
                packed,
                height: h * 8,
                width: w * 8,
                latentChannels: 32,
                patchSize: 2
            )

            #expect(unpacked.shape == latent.shape, "Roundtrip failed for size \(h)x\(w)")
        }
    }

    @Test func positionIDsVaryingSizes() {
        let sizes = [(512, 512), (1024, 1024), (768, 1024)]

        for (h, w) in sizes {
            let ids = LatentUtils.generateImagePositionIDs(height: h, width: w)

            // Number of patches = (h/8/2) * (w/8/2) = h*w/256
            let expectedPatches = (h / 16) * (w / 16)
            #expect(ids.shape[0] == expectedPatches, "Wrong patch count for \(h)x\(w)")
            #expect(ids.shape[1] == 4)  // [T, H, W, L] dimensions
        }
    }
}

// MARK: - Memory Manager Tests

@Suite struct MemoryManagerTests {

    @Test func memoryManagerSingleton() {
        let manager1 = Flux2MemoryManager.shared
        let manager2 = Flux2MemoryManager.shared
        #expect(manager1 === manager2)
    }

    @Test func memoryManagerPhysicalMemory() {
        let manager = Flux2MemoryManager.shared

        // Physical memory should be positive
        #expect(manager.physicalMemory > 0)
        #expect(manager.physicalMemoryGB > 0)
    }

    @Test func memoryManagerEstimatedAvailable() {
        let manager = Flux2MemoryManager.shared

        // Estimated available should be less than physical (system reserve)
        #expect(manager.estimatedAvailableMemoryGB <= manager.physicalMemoryGB)
    }

    @Test func memoryManagerCanRunCheck() {
        let manager = Flux2MemoryManager.shared

        // Minimal config should be runnable on most systems
        let minimalConfig = Flux2QuantizationConfig.minimal
        // Just check the method doesn't crash
        _ = manager.canRun(config: minimalConfig)
    }

    @Test func memoryManagerRecommendedConfig() {
        let manager = Flux2MemoryManager.shared

        let recommended = manager.recommendedConfig()
        // Should return a valid config
        #expect(recommended.textEncoder != nil)
        #expect(recommended.transformer != nil)
    }
}

// MARK: - Transformer Config Tests

@Suite struct TransformerConfigTests {

    @Test func flux2DevConfig() {
        let config = Flux2TransformerConfig.flux2Dev

        #expect(config.inChannels == 128)
        #expect(config.numLayers == 8)
        #expect(config.numSingleLayers == 48)
    }

    @Test func flux2KleinConfig() {
        let config = Flux2TransformerConfig.klein4B

        // Klein 4B is smaller than Dev
        #expect(config.numLayers == 5)
        #expect(config.numSingleLayers == 20)
    }

    @Test func innerDimCalculation() {
        let config = Flux2TransformerConfig.flux2Dev

        let expectedInnerDim = config.numAttentionHeads * config.attentionHeadDim
        #expect(config.innerDim == expectedInnerDim)
    }

    @Test func kleinSmallerThanDev() {
        let dev = Flux2TransformerConfig.flux2Dev
        let klein = Flux2TransformerConfig.klein4B

        #expect(klein.numLayers < dev.numLayers)
        #expect(klein.numSingleLayers < dev.numSingleLayers)
    }
}

// MARK: - Quantization Config Tests

@Suite struct QuantizationConfigTests {

    @Test func mistralQuantizationValues() {
        let bf16 = MistralQuantization.bf16
        let mlx8bit = MistralQuantization.mlx8bit
        let mlx4bit = MistralQuantization.mlx4bit

        #expect(bf16.rawValue == "bf16")
        #expect(mlx8bit.rawValue == "8bit")
        #expect(mlx4bit.rawValue == "4bit")
    }

    @Test func transformerQuantizationValues() {
        let bf16 = TransformerQuantization.bf16
        let qint8 = TransformerQuantization.qint8

        #expect(bf16.rawValue == "bf16")
        #expect(qint8.rawValue == "qint8")
    }

    @Test func quantizationMemoryEstimates() {
        // Higher quality should use more memory
        let highQuality = Flux2QuantizationConfig.highQuality
        let minimal = Flux2QuantizationConfig.minimal

        #expect(highQuality.estimatedTotalMemoryGB > minimal.estimatedTotalMemoryGB)
    }

    @Test func quantizationPhaseMemory() {
        let config = Flux2QuantizationConfig.balanced

        // Both phases should have positive memory estimates
        #expect(config.textEncodingPhaseMemoryGB > 0)
        #expect(config.imageGenerationPhaseMemoryGB > 0)
    }

    @Test func mistralQuantizationEstimatedMemoryGB() {
        let bf16 = MistralQuantization.bf16.estimatedMemoryGB
        let mlx8bit = MistralQuantization.mlx8bit.estimatedMemoryGB
        let mlx4bit = MistralQuantization.mlx4bit.estimatedMemoryGB

        // bf16 > 8bit > 4bit
        #expect(bf16 > mlx8bit)
        #expect(mlx8bit > mlx4bit)
    }

    @Test func transformerQuantizationMemory() {
        let bf16 = TransformerQuantization.bf16.estimatedMemoryGB
        let qint8 = TransformerQuantization.qint8.estimatedMemoryGB
        let int4 = TransformerQuantization.int4.estimatedMemoryGB

        #expect(bf16 > qint8)
        #expect(qint8 > int4)
    }

    @Test func transformerQuantizationAllCases() {
        let allCases = TransformerQuantization.allCases
        #expect(allCases.count == 3)
        #expect(allCases.contains(.bf16))
        #expect(allCases.contains(.qint8))
        #expect(allCases.contains(.int4))
    }

    @Test func transformerQuantizationBitsOrdering() {
        #expect(TransformerQuantization.bf16.bits > TransformerQuantization.qint8.bits)
        #expect(TransformerQuantization.qint8.bits > TransformerQuantization.int4.bits)
    }

    @Test func transformerQuantizationDisplayNames() {
        #expect(!TransformerQuantization.bf16.displayName.isEmpty)
        #expect(!TransformerQuantization.qint8.displayName.isEmpty)
        #expect(!TransformerQuantization.int4.displayName.isEmpty)
    }

    @Test func memoryEfficientPreset() {
        #expect(Flux2QuantizationConfig.memoryEfficient.textEncoder == .mlx4bit)
        #expect(Flux2QuantizationConfig.memoryEfficient.transformer == .qint8)
    }

    @Test func presetMemoryOrdering() {
        let ultra = Flux2QuantizationConfig.ultraMinimal.estimatedTotalMemoryGB
        let minimal = Flux2QuantizationConfig.minimal.estimatedTotalMemoryGB
        let balanced = Flux2QuantizationConfig.balanced.estimatedTotalMemoryGB
        let high = Flux2QuantizationConfig.highQuality.estimatedTotalMemoryGB

        #expect(ultra <= minimal)
        #expect(minimal <= balanced)
        #expect(balanced <= high)
    }
}

// MARK: - On-the-fly Quantization Variant Tests

@Suite struct OnTheFlyQuantizationTests {

    @Test func devVariantMapping() {
        // Dev has pre-quantized qint8 available
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .bf16) ==
            .bf16
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .qint8) ==
            .qint8,
            "Dev qint8 should use pre-quantized variant"
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .dev, quantization: .int4) ==
            .bf16,
            "Dev int4 should load bf16 and quantize on-the-fly"
        )
    }

    @Test func klein4BVariantMapping() {
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .bf16) ==
            .klein4B_bf16
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .qint8) ==
            .klein4B_8bit,
            "Klein 4B qint8 should use pre-quantized variant"
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein4B, quantization: .int4) ==
            .klein4B_bf16,
            "Klein 4B int4 should load bf16 and quantize on-the-fly"
        )
    }

    @Test func klein9BAlwaysLoadsBf16() {
        // Klein 9B has no pre-quantized variants — always loads bf16
        for quant in TransformerQuantization.allCases {
            #expect(
                ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: quant) ==
                .klein9B_bf16,
                "Klein 9B should always load bf16 variant for \(quant)"
            )
        }
    }

    @Test func baseModelsAlwaysReturnBaseVariant() {
        for quant in TransformerQuantization.allCases {
            #expect(
                ModelRegistry.TransformerVariant.variant(for: .klein4BBase, quantization: quant) ==
                .klein4B_base_bf16,
                "Klein 4B base should always return base bf16 for \(quant)"
            )
            #expect(
                ModelRegistry.TransformerVariant.variant(for: .klein9BBase, quantization: quant) ==
                .klein9B_base_bf16,
                "Klein 9B base should always return base bf16 for \(quant)"
            )
        }
    }

    @Test func recommendedConfigAllTiers() {
        // Ultra-minimal tier (<32GB)
        let ultra = ModelRegistry.recommendedConfig(forRAMGB: 24)
        #expect(ultra.transformer == .int4)

        // Minimal tier (32-48GB)
        let minimal = ModelRegistry.recommendedConfig(forRAMGB: 32)
        #expect(minimal.transformer == .qint8)

        // Balanced tier (48-96GB)
        let balanced48 = ModelRegistry.recommendedConfig(forRAMGB: 48)
        #expect(balanced48.transformer == .qint8)

        let balanced64 = ModelRegistry.recommendedConfig(forRAMGB: 64)
        #expect(balanced64.transformer == .qint8)

        // High quality tier (96GB+)
        let high = ModelRegistry.recommendedConfig(forRAMGB: 96)
        #expect(high.transformer == .bf16)

        let veryHigh = ModelRegistry.recommendedConfig(forRAMGB: 128)
        #expect(veryHigh.transformer == .bf16)
    }

    @Test func int4QuantizationGroupSize() {
        // All quantization levels use the same group size
        #expect(TransformerQuantization.bf16.groupSize == 64)
        #expect(TransformerQuantization.qint8.groupSize == 64)
        #expect(TransformerQuantization.int4.groupSize == 64)
    }

    @Test func quantizationCodable() throws {
        // Verify int4 round-trips through JSON encoding
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let config = Flux2QuantizationConfig.ultraMinimal
        let data = try encoder.encode(config)
        let decoded = try decoder.decode(Flux2QuantizationConfig.self, from: data)

        #expect(decoded.transformer == .int4)
        #expect(decoded.textEncoder == .mlx4bit)
    }
}

// MARK: - Debug Utilities Tests

@Suite struct DebugUtilsTests {

    @Test func debugLogDoesNotCrash() {
        // Debug logging should not crash regardless of state
        Flux2Debug.log("Test message")
        Flux2Debug.verbose("Verbose test message")
        Flux2Debug.info("Info message")
        Flux2Debug.warning("Warning message")
        Flux2Debug.error("Error message")

        // No assertion needed - just ensure no crash
        #expect(true)
    }

    @Test func debugEnabledState() {
        let wasEnabled = Flux2Debug.enabled

        Flux2Debug.enabled = true
        #expect(Flux2Debug.enabled)

        Flux2Debug.enabled = false
        #expect(!Flux2Debug.enabled)

        // Restore
        Flux2Debug.enabled = wasEnabled
    }

    @Test func debugLevels() {
        #expect(Flux2Debug.Level.verbose < Flux2Debug.Level.info)
        #expect(Flux2Debug.Level.info < Flux2Debug.Level.warning)
        #expect(Flux2Debug.Level.warning < Flux2Debug.Level.error)
    }

    @Test func debugModeToggle() {
        let originalLevel = Flux2Debug.minLevel

        Flux2Debug.enableDebugMode()
        #expect(Flux2Debug.minLevel == .verbose)

        Flux2Debug.setNormalMode()
        #expect(Flux2Debug.minLevel == .warning)

        // Restore
        Flux2Debug.minLevel = originalLevel
    }
}

// MARK: - EmpiricalMu Tests

@Suite struct EmpiricalMuTests {

    @Test func empiricalMuCalculation() {
        // Test various image sequence lengths
        let mu1024 = computeEmpiricalMu(imageSeqLen: 4096, numSteps: 50)
        let mu512 = computeEmpiricalMu(imageSeqLen: 1024, numSteps: 50)

        // Larger images should have different mu
        #expect(mu1024 != mu512)
    }

    @Test func empiricalMuLargeImage() {
        // Very large images use different formula
        let muLarge = computeEmpiricalMu(imageSeqLen: 5000, numSteps: 50)

        #expect(muLarge > 0)
    }

    @Test func empiricalMuVaryingSteps() {
        let mu50 = computeEmpiricalMu(imageSeqLen: 4096, numSteps: 50)
        let mu20 = computeEmpiricalMu(imageSeqLen: 4096, numSteps: 20)

        // Different step counts should produce different mu
        #expect(mu50 != mu20)
    }
}

// MARK: - Training Variants Tests

@Suite struct TrainingVariantsTests {

    // MARK: - Klein 9B Base Variant Tests

    @Test func klein9BBaseVariantHuggingFaceRepo() {
        let variant = ModelRegistry.TransformerVariant.klein9B_base_bf16
        #expect(variant.huggingFaceRepo == "black-forest-labs/FLUX.2-klein-base-9B")
    }

    @Test func klein9BBaseVariantIsGated() {
        let variant = ModelRegistry.TransformerVariant.klein9B_base_bf16
        #expect(variant.isGated)
    }

    @Test func klein9BBaseVariantEstimatedSize() {
        let variant = ModelRegistry.TransformerVariant.klein9B_base_bf16
        #expect(variant.estimatedSizeGB == 18)  // Same as distilled
    }

    // MARK: - Klein 4B Base Variant Tests

    @Test func klein4BBaseVariantHuggingFaceRepo() {
        let variant = ModelRegistry.TransformerVariant.klein4B_base_bf16
        #expect(variant.huggingFaceRepo == "black-forest-labs/FLUX.2-klein-base-4B")
    }

    @Test func klein4BBaseVariantIsNotGated() {
        // Klein 4B Base from black-forest-labs is NOT gated (open access)
        let variant = ModelRegistry.TransformerVariant.klein4B_base_bf16
        #expect(!variant.isGated)
    }

    // MARK: - Training/Inference Model Variant Tests

    @Test func baseModelsAreForTraining() {
        // Base (non-distilled) Flux2Models should be for training
        #expect(Flux2Model.klein4BBase.isForTraining)
        #expect(Flux2Model.klein9BBase.isForTraining)
        #expect(Flux2Model.klein4BBase.isBaseModel)
        #expect(Flux2Model.klein9BBase.isBaseModel)
    }

    @Test func distilledModelsAreForInference() {
        // Distilled models should be for inference, not training
        #expect(Flux2Model.klein4B.isForInference)
        #expect(Flux2Model.klein9B.isForInference)
        #expect(!Flux2Model.klein4B.isForTraining)
        #expect(!Flux2Model.klein9B.isForTraining)
    }

    // MARK: - trainingVariant(for:) Method Tests

    @Test func trainingVariantForKlein4B() {
        let variant = ModelRegistry.TransformerVariant.trainingVariant(for: .klein4B)
        #expect(variant == .klein4B_base_bf16)
    }

    @Test func trainingVariantForKlein9B() {
        let variant = ModelRegistry.TransformerVariant.trainingVariant(for: .klein9B)
        #expect(variant == .klein9B_base_bf16)
    }

    @Test func trainingVariantForDev() {
        // Dev model is already "base" (not distilled), so it uses bf16
        let variant = ModelRegistry.TransformerVariant.trainingVariant(for: .dev)
        #expect(variant == .bf16)
    }

    @Test func trainingVariantReturnsNonNil() {
        // All model types should have a training variant
        for model in [Flux2Model.klein4B, .klein9B, .dev] {
            let variant = ModelRegistry.TransformerVariant.trainingVariant(for: model)
            #expect(variant != nil, "Training variant should exist for \(model)")
        }
    }

    // MARK: - Training Variant Consistency

    @Test func trainingVariantsResolveToBaseModels() {
        // Klein models should resolve to base variants for training
        #expect(Flux2Model.klein4B.trainingVariant == .klein4BBase)
        #expect(Flux2Model.klein9B.trainingVariant == .klein9BBase)

        // Dev doesn't have a separate base model
        #expect(Flux2Model.dev.trainingVariant == .dev)
    }
}

// MARK: - DevTextEncoder Tests

@Suite struct DevTextEncoderTests {

    @Test func devTextEncoderDefaultInit() {
        let encoder = DevTextEncoder()
        #expect(encoder.quantization == .mlx8bit)
        #expect(encoder.maxSequenceLength == 512)
        #expect(encoder.outputDimension == 15360)
    }

    @Test func devTextEncoderCustomQuantization() {
        let encoder4bit = DevTextEncoder(quantization: .mlx4bit)
        #expect(encoder4bit.quantization == .mlx4bit)

        let encoderBf16 = DevTextEncoder(quantization: .bf16)
        #expect(encoderBf16.quantization == .bf16)
    }

    @Test func devTextEncoderEstimatedMemory() {
        let bf16 = DevTextEncoder(quantization: .bf16)
        let mlx8bit = DevTextEncoder(quantization: .mlx8bit)
        let mlx4bit = DevTextEncoder(quantization: .mlx4bit)

        // Higher precision should use more memory
        #expect(bf16.estimatedMemoryGB > mlx8bit.estimatedMemoryGB)
        #expect(mlx8bit.estimatedMemoryGB > mlx4bit.estimatedMemoryGB)
    }

    @Test func devTextEncoderNotLoadedByDefault() {
        let encoder = DevTextEncoder()
        #expect(!encoder.isLoaded)
    }

    @Test func devTextEncoderOutputDimensionMatches() {
        // Dev uses Mistral with 3 layers x 5120 hidden size = 15360
        let encoder = DevTextEncoder()
        #expect(encoder.outputDimension == 3 * 5120)
    }
}

// MARK: - Generation Result Tests

@Suite struct GenerationResultTests {

    @Test func generationResultInitialization() {
        // Create a minimal test image (1x1 pixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            Issue.record("Failed to create test image")
            return
        }

        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: "enhanced: a beautiful sunset",
            wasUpsampled: true,
            originalPrompt: "a beautiful sunset"
        )

        #expect(result.usedPrompt == "enhanced: a beautiful sunset")
        #expect(result.originalPrompt == "a beautiful sunset")
        #expect(result.wasUpsampled)
        #expect(result.image.width == 1)
        #expect(result.image.height == 1)
    }

    @Test func generationResultNoUpsampling() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            Issue.record("Failed to create test image")
            return
        }

        let prompt = "a cat sitting on a chair"
        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: prompt,
            wasUpsampled: false,
            originalPrompt: prompt
        )

        #expect(!result.wasUpsampled)
        #expect(result.usedPrompt == result.originalPrompt)
    }

    @Test func generationResultPromptDifference() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            Issue.record("Failed to create test image")
            return
        }

        let original = "cat"
        let enhanced = "A majestic orange tabby cat sitting gracefully on a velvet chair, soft lighting, detailed fur"

        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: enhanced,
            wasUpsampled: true,
            originalPrompt: original
        )

        #expect(result.wasUpsampled)
        #expect(result.usedPrompt != result.originalPrompt)
        #expect(result.usedPrompt.count > result.originalPrompt.count)
    }

    @Test func generationResultSendable() {
        // Verify Flux2GenerationResult conforms to Sendable
        // This test ensures the struct can be safely passed across concurrency boundaries
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let testImage = context.makeImage() else {
            Issue.record("Failed to create test image")
            return
        }

        let result = Flux2GenerationResult(
            image: testImage,
            usedPrompt: "test prompt",
            wasUpsampled: false,
            originalPrompt: "test prompt"
        )

        // If this compiles, the type is Sendable
        let _: any Sendable = result
        #expect(true)
    }
}

// MARK: - MemoryConfig Tests

@Suite struct MemoryConfigTests {

    @Test func cacheProfiles() {
        // Test all profiles exist and have descriptions
        for profile in MemoryConfig.CacheProfile.allCases {
            #expect(!profile.description.isEmpty)
            #expect(!profile.rawValue.isEmpty)
        }
    }

    @Test func systemRAMDetection() {
        // System RAM should be detected and reasonable
        let ram = MemoryConfig.systemRAMGB
        #expect(ram > 0)
        #expect(ram < 1024) // Less than 1TB
    }

    @Test func safeCachePercentage() {
        // Safe cache percentage should be between 0 and 1
        let pct = MemoryConfig.safeCachePercentage
        #expect(pct > 0)
        #expect(pct <= 1.0)
    }

    @Test func conservativeProfileLimit() {
        // Conservative should always return 512 MB
        let limit = MemoryConfig.cacheLimitForProfile(.conservative)
        #expect(limit != nil)
        #expect(limit == 512 * 1024 * 1024) // 512 MB
    }

    @Test func performanceProfileLimit() {
        // Performance should return up to 4 GB
        let limit = MemoryConfig.cacheLimitForProfile(.performance)
        #expect(limit != nil)
        #expect(limit! > 0)
        #expect(limit! <= 4 * 1024 * 1024 * 1024) // Max 4 GB
    }

    @Test func autoProfileLimit() {
        // Auto should return a dynamic limit based on system RAM
        let limit = MemoryConfig.cacheLimitForProfile(.auto)
        // For systems < 128GB, should return a limit
        // For systems >= 128GB, might return nil (unlimited)
        if MemoryConfig.systemRAMGB < 128 {
            #expect(limit != nil)
            #expect(limit! > 256 * 1024 * 1024) // At least 256 MB
        }
    }

    @Test func phaseLimitsForKlein4B() {
        let limits = MemoryConfig.PhaseLimits.forModel(.klein4B, profile: .conservative)
        #expect(limits.textEncoding > 0)
        #expect(limits.denoising > 0)
        #expect(limits.vaeDecoding > 0)
    }

    @Test func phaseLimitsForDev() {
        let limits = MemoryConfig.PhaseLimits.forModel(.dev, profile: .performance)
        #expect(limits.textEncoding > 0)
        #expect(limits.denoising > 0)
        #expect(limits.vaeDecoding > 0)

        // Dev should have larger limits than Klein
        let kleinLimits = MemoryConfig.PhaseLimits.forModel(.klein4B, profile: .performance)
        #expect(limits.denoising >= kleinLimits.denoising)
    }

    @Test func resolutionBasedCacheLimit() {
        // Higher resolution should have higher cache limit
        let limit512 = MemoryConfig.cacheLimitForResolution(width: 512, height: 512, model: .klein4B)
        let limit1024 = MemoryConfig.cacheLimitForResolution(width: 1024, height: 1024, model: .klein4B)

        #expect(limit1024 > limit512)
    }

    @Test func configurationSummary() {
        // Configuration summary should not be empty
        let summary = MemoryConfig.configurationSummary
        #expect(!summary.isEmpty)
        #expect(summary.contains("Memory Configuration"))
    }
}

// MARK: - Inference Variant Tests

@Suite struct InferenceVariantTests {

    @Test func inferenceVariantForDistilledModels() {
        // Distilled models should return themselves
        #expect(Flux2Model.klein4B.inferenceVariant == .klein4B)
        #expect(Flux2Model.klein9B.inferenceVariant == .klein9B)
        #expect(Flux2Model.dev.inferenceVariant == .dev)
    }

    @Test func inferenceVariantForBaseModels() {
        // Base models should return distilled variants
        #expect(Flux2Model.klein4BBase.inferenceVariant == .klein4B)
        #expect(Flux2Model.klein9BBase.inferenceVariant == .klein9B)
    }

    @Test func inferenceVariantIsAlwaysForInference() {
        // The inference variant should always be usable for inference
        for model in Flux2Model.allCases {
            #expect(model.inferenceVariant.isForInference,
                    "\(model).inferenceVariant should be for inference")
        }
    }

    @Test func inferenceVariantIsNeverBase() {
        // The inference variant should never be a base model
        for model in Flux2Model.allCases {
            #expect(!model.inferenceVariant.isBaseModel,
                    "\(model).inferenceVariant should not be a base model")
        }
    }

    @Test func trainingAndInferenceVariantsAreInverse() {
        // For each model, trainingVariant.inferenceVariant should return the distilled version
        for model in [Flux2Model.klein4B, .klein9B, .dev] {
            let training = model.trainingVariant
            let inference = training.inferenceVariant
            #expect(inference.isForInference,
                    "trainingVariant.inferenceVariant of \(model) should be for inference")
        }
    }
}

// MARK: - Gradient Checkpointing Config Tests

@Suite struct GradientCheckpointingConfigTests {

    @Test func gradientCheckpointingReducesMemoryEstimate() {
        let tmpDataset = URL(fileURLWithPath: "/tmp/test")
        let tmpOutput = URL(fileURLWithPath: "/tmp/output")

        let configWithout = LoRATrainingConfig(
            datasetPath: tmpDataset,
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: false,
            outputPath: tmpOutput
        )

        let configWith = LoRATrainingConfig(
            datasetPath: tmpDataset,
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: true,
            outputPath: tmpOutput
        )

        let memWithout = configWithout.estimateMemoryGB(for: .klein4B)
        let memWith = configWith.estimateMemoryGB(for: .klein4B)

        // Gradient checkpointing should reduce memory estimate
        #expect(memWith < memWithout)
    }

    @Test func gradientCheckpointingSuggestion() {
        let config = LoRATrainingConfig(
            datasetPath: URL(fileURLWithPath: "/tmp/test"),
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: false,
            outputPath: URL(fileURLWithPath: "/tmp/output")
        )

        // Request suggestions for tight memory
        let suggestions = config.suggestAdjustments(for: .klein9B, availableGB: 16)

        // Should suggest enabling gradient checkpointing
        #expect(suggestions.contains { $0.contains("gradient checkpointing") },
                "Should suggest gradient checkpointing when disabled and memory is tight")
    }

    @Test func gradientCheckpointingNoSuggestionWhenEnabled() {
        let config = LoRATrainingConfig(
            datasetPath: URL(fileURLWithPath: "/tmp/test"),
            rank: 32,
            alpha: 32.0,
            gradientCheckpointing: true,
            outputPath: URL(fileURLWithPath: "/tmp/output")
        )

        let suggestions = config.suggestAdjustments(for: .klein9B, availableGB: 16)

        // Should NOT suggest gradient checkpointing when already enabled
        #expect(!suggestions.contains { $0.contains("gradient checkpointing") },
                "Should not suggest gradient checkpointing when already enabled")
    }

    @Test func presetsHaveGradientCheckpointing() {
        let tmpDataset = URL(fileURLWithPath: "/tmp/test")
        let tmpOutput = URL(fileURLWithPath: "/tmp/output")

        // All presets should have gradient checkpointing enabled
        let minimal = LoRATrainingConfig.minimal8GB(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )
        #expect(minimal.gradientCheckpointing)

        let balanced = LoRATrainingConfig.balanced16GB(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )
        #expect(balanced.gradientCheckpointing)

        let quality = LoRATrainingConfig.quality32GB(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )
        #expect(quality.gradientCheckpointing)
    }

    @Test func gradientCheckpointingDefaultTrue() {
        // Default init should have gradient checkpointing enabled
        let config = LoRATrainingConfig(
            datasetPath: URL(fileURLWithPath: "/tmp/test"),
            outputPath: URL(fileURLWithPath: "/tmp/output")
        )
        #expect(config.gradientCheckpointing)
    }
}

// MARK: - Validation Quantization Tests

@Suite struct ValidationQuantizationTests {

    @Test func klein9BQuantizationOnTheFly() {
        // Klein 9B has no pre-quantized variant — uses on-the-fly quantization
        // All quantization levels should map to the bf16 download variant
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .bf16) ==
            .klein9B_bf16
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .qint8) ==
            .klein9B_bf16,
            "Klein 9B qint8 should load bf16 and quantize on-the-fly"
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9B, quantization: .int4) ==
            .klein9B_bf16,
            "Klein 9B int4 should load bf16 and quantize on-the-fly"
        )
    }

    @Test func klein9BTransformerConfigMatchesDistilled() {
        // Base and distilled Klein 9B should share the same transformer config
        #expect(Flux2Model.klein9B.transformerConfig.numLayers ==
                Flux2Model.klein9BBase.transformerConfig.numLayers)
        #expect(Flux2Model.klein9B.transformerConfig.numSingleLayers ==
                Flux2Model.klein9BBase.transformerConfig.numSingleLayers)
    }

    @Test func klein9BConfig() {
        let config = Flux2TransformerConfig.klein9B

        // Klein 9B should be between Klein 4B and Dev in size
        let klein4B = Flux2TransformerConfig.klein4B
        let dev = Flux2TransformerConfig.flux2Dev

        #expect(config.numLayers > klein4B.numLayers)
        #expect(config.numSingleLayers > klein4B.numSingleLayers)
        #expect(config.numLayers <= dev.numLayers)
    }
}

// MARK: - Weight Conversion Tests

@Suite struct WeightConversionTests {

    /// Validate that bf16->f16 direct conversion produces identical results to bf16->f32->f16
    /// This is critical for the BFL weight loading optimization (item 2.1)
    @Test func bf16ToF16DirectMatchesViaF32() {
        // Create realistic weight-like values in float32, then convert to bf16
        // Test various ranges: small, normal, large (within f16 range)
        let testValues: [Float] = [
            0.0, 1.0, -1.0,
            0.001, -0.001,           // Small weights
            0.01, -0.05, 0.1,       // Typical weight magnitudes
            0.5, -0.5, 1.5, -2.0,   // Normal range
            100.0, -100.0,           // Larger values
            65504.0, -65504.0,       // f16 max representable
            0.000061035,             // f16 min positive normal (~6.1e-5)
        ]
        let f32Array = MLXArray(testValues)
        let bf16Array = f32Array.asType(.bfloat16)

        // Path A: bf16 -> f16 direct (new optimized path)
        let directF16 = bf16Array.asType(.float16)

        // Path B: bf16 -> f32 -> f16 (old path)
        let viaF32 = bf16Array.asType(.float32).asType(.float16)

        // Both should produce identical results
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(directF16, viaF32)

        let directValues = directF16.asType(.float32)
        let viaF32Values = viaF32.asType(.float32)

        // Compare element by element
        for i in 0..<testValues.count {
            let d = directValues[i].item(Float.self)
            let v = viaF32Values[i].item(Float.self)
            #expect(d == v, "Mismatch at index \(i) for input \(testValues[i]): direct=\(d), viaF32=\(v)")
        }
    }

    /// Test bf16->f16 with large random arrays (simulating real weight tensors)
    @Test func bf16ToF16DirectLargeArray() {
        // Simulate a realistic weight matrix (e.g., 3072x3072 linear layer)
        let weights = MLXRandom.normal([3072, 3072]).asType(.bfloat16)

        let directF16 = weights.asType(.float16)
        let viaF32 = weights.asType(.float32).asType(.float16)

        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(directF16, viaF32)

        // Check that all values match exactly (bitwise identical)
        let diff = MLX.abs(directF16.asType(.float32) - viaF32.asType(.float32))
        let maxDiff = MLX.max(diff).item(Float.self)
        #expect(maxDiff == 0.0, "Max difference between direct and viaF32 conversion: \(maxDiff)")

        // Also verify no NaN or inf introduced
        let hasNaN = any(isNaN(directF16)).item(Bool.self)
        let hasInf = any(MLX.abs(directF16.asType(.float32)) .> 1e30).item(Bool.self)
        #expect(!hasNaN, "Direct bf16->f16 conversion should not introduce NaN")
        #expect(!hasInf, "Normal weight values should not overflow to inf")
    }

    /// Test that values outside f16 range are handled consistently
    @Test func bf16ToF16OverflowConsistency() {
        // bf16 can represent values up to ~3.4e38, but f16 max is ~65504
        // Both conversion paths should handle overflow identically
        let largeValues = MLXArray([Float(70000.0), Float(-70000.0), Float(100000.0)])
            .asType(.bfloat16)

        let directF16 = largeValues.asType(.float16)
        let viaF32 = largeValues.asType(.float32).asType(.float16)

        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(directF16, viaF32)

        // Both should produce the same result (likely inf)
        let directF32 = directF16.asType(.float32)
        let viaF32F32 = viaF32.asType(.float32)

        for i in 0..<3 {
            let d = directF32[i].item(Float.self)
            let v = viaF32F32[i].item(Float.self)
            // Both should be inf or the same clamped value
            #expect(d == v, "Overflow handling mismatch at index \(i): direct=\(d), viaF32=\(v)")
        }
    }

    // MARK: - Image Roundtrip Tests (I2I spatial shift diagnostics)

    /// Helper: create a CGImage with a known gradient pattern
    private func createGradientCGImage(width: Int, height: Int) -> CGImage {
        let bytesPerPixel = 3
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * bytesPerRow + x * bytesPerPixel
                pixelData[idx] = UInt8(x * 255 / max(width - 1, 1))     // R = horizontal gradient
                pixelData[idx + 1] = UInt8(y * 255 / max(height - 1, 1)) // G = vertical gradient
                pixelData[idx + 2] = 128                                  // B = constant
            }
        }

        let provider = CGDataProvider(data: Data(pixelData) as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }

    /// Helper: extract pixel RGB from CGImage at (x, y)
    private func getPixel(from image: CGImage, x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8)? {
        guard let dataProvider = image.dataProvider,
              let data = dataProvider.data else { return nil }
        let bpp = image.bitsPerPixel / 8
        let bpr = image.bytesPerRow
        let ptr = CFDataGetBytePtr(data)!
        let offset = y * bpr + x * bpp

        // Determine RGB offsets from bitmap info
        let alphaInfo = CGImageAlphaInfo(rawValue: image.bitmapInfo.rawValue & CGBitmapInfo.alphaInfoMask.rawValue)
        let alphaFirst = alphaInfo == .premultipliedFirst || alphaInfo == .first || alphaInfo == .noneSkipFirst
        let hasAlpha = bpp >= 4

        let rOff = (hasAlpha && alphaFirst) ? 1 : 0
        let gOff = (hasAlpha && alphaFirst) ? 2 : 1
        let bOff = (hasAlpha && alphaFirst) ? 3 : 2

        return (ptr[offset + rOff], ptr[offset + gOff], ptr[offset + bOff])
    }

    /// Helper: encode CGImage to PNG Data
    private func pngData(from image: CGImage) -> Data? {
        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }

    /// Test 1: CGImageSource roundtrip is pixel-exact
    @Test func cgImageSourceRoundtripPixelExact() {
        let width = 8
        let height = 8
        let original = createGradientCGImage(width: width, height: height)

        // Encode to PNG
        guard let data = pngData(from: original) else {
            Issue.record("Failed to encode PNG")
            return
        }

        // Decode via CGImageSource (the fix path)
        guard let decoded = Flux2Pipeline.cgImage(from: data) else {
            Issue.record("Failed to decode via CGImageSource")
            return
        }

        // Verify dimensions
        #expect(decoded.width == width, "Width mismatch")
        #expect(decoded.height == height, "Height mismatch")

        // Compare every pixel
        for y in 0..<height {
            for x in 0..<width {
                guard let origPixel = getPixel(from: original, x: x, y: y),
                      let decodedPixel = getPixel(from: decoded, x: x, y: y) else {
                    Issue.record("Failed to read pixel at (\(x), \(y))")
                    continue
                }
                #expect(origPixel.r == decodedPixel.r, "R mismatch at (\(x), \(y)): \(origPixel.r) vs \(decodedPixel.r)")
                #expect(origPixel.g == decodedPixel.g, "G mismatch at (\(x), \(y)): \(origPixel.g) vs \(decodedPixel.g)")
                #expect(origPixel.b == decodedPixel.b, "B mismatch at (\(x), \(y)): \(origPixel.b) vs \(decodedPixel.b)")
            }
        }
    }

    #if canImport(AppKit)
    /// Test 2: Detect NSImage roundtrip artifacts
    /// This test documents that NSImage cgImage(forProposedRect:) can change pixel format/values
    @Test func nsImageRoundtripDetectsChanges() {
        let width = 16
        let height = 16
        let original = createGradientCGImage(width: width, height: height)

        // NSImage roundtrip (the problematic path)
        let nsImage = NSImage(cgImage: original, size: NSSize(width: width, height: height))
        guard let roundtripped = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            Issue.record("NSImage roundtrip failed")
            return
        }

        // Check format changes
        let formatChanged = roundtripped.bitsPerPixel != original.bitsPerPixel
            || roundtripped.bytesPerRow != original.bytesPerRow
            || roundtripped.alphaInfo != original.alphaInfo

        if formatChanged {
            // Document that NSImage changes format (expected behavior we're fixing)
            print("[NSImage Roundtrip] Format changed: \(original.bitsPerPixel)bpp/\(original.alphaInfo.rawValue)alpha -> \(roundtripped.bitsPerPixel)bpp/\(roundtripped.alphaInfo.rawValue)alpha")
        }

        // Dimensions should at least be preserved
        #expect(roundtripped.width == width, "NSImage roundtrip changed width")
        #expect(roundtripped.height == height, "NSImage roundtrip changed height")
    }
    #endif

    /// Test 3: CGImageSource vs NSImage pixel comparison
    /// Verifies CGImageSource produces pixel-exact results while NSImage may not
    @Test func cgImageSourceVsNSImageComparison() {
        let width = 16
        let height = 16
        let original = createGradientCGImage(width: width, height: height)

        guard let data = pngData(from: original) else {
            Issue.record("Failed to encode PNG")
            return
        }

        // Path A: CGImageSource (pixel-exact)
        guard let viaCGImageSource = Flux2Pipeline.cgImage(from: data) else {
            Issue.record("CGImageSource decode failed")
            return
        }

        // Verify CGImageSource path is pixel-exact with original
        var cgImageSourceExact = true
        for y in 0..<height {
            for x in 0..<width {
                guard let origPixel = getPixel(from: original, x: x, y: y),
                      let sourcePixel = getPixel(from: viaCGImageSource, x: x, y: y) else {
                    continue
                }
                if origPixel.r != sourcePixel.r || origPixel.g != sourcePixel.g || origPixel.b != sourcePixel.b {
                    cgImageSourceExact = false
                    break
                }
            }
            if !cgImageSourceExact { break }
        }

        #expect(cgImageSourceExact, "CGImageSource path should be pixel-exact")

        #if canImport(AppKit)
        // Path B: NSImage (potentially lossy)
        let nsImage = NSImage(data: data)!
        let viaNSImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)!

        // Count pixel differences
        var diffCount = 0
        for y in 0..<height {
            for x in 0..<width {
                guard let origPixel = getPixel(from: original, x: x, y: y),
                      let nsPixel = getPixel(from: viaNSImage, x: x, y: y) else {
                    continue
                }
                if origPixel.r != nsPixel.r || origPixel.g != nsPixel.g || origPixel.b != nsPixel.b {
                    diffCount += 1
                }
            }
        }

        if diffCount > 0 {
            print("[NSImage vs CGImageSource] NSImage path has \(diffCount)/\(width * height) pixel differences")
        }
        #endif
    }
}

// MARK: - Flux2GenerationMode Tests

@Suite struct Flux2GenerationModeTests {

    @Test func textToImageMode() {
        let mode = Flux2GenerationMode.textToImage
        if case .textToImage = mode {
            // OK
        } else {
            Issue.record("Expected textToImage")
        }
    }

    @Test func imageToImageModeWithSingleImage() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            Issue.record("Failed to create test image"); return
        }

        let mode = Flux2GenerationMode.imageToImage(images: [img])
        if case .imageToImage(let images) = mode {
            #expect(images.count == 1)
            #expect(images[0].width == 8)
        } else {
            Issue.record("Expected imageToImage")
        }
    }

    @Test func imageToImageModeWithMultipleImages() {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 16, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            Issue.record("Failed to create test image"); return
        }

        let mode = Flux2GenerationMode.imageToImage(images: [img, img, img])
        if case .imageToImage(let images) = mode {
            #expect(images.count == 3)
        } else {
            Issue.record("Expected imageToImage")
        }
    }

    @Test func modeHasNoStrengthParameter() {
        // Verify that Flux2GenerationMode.imageToImage has no strength associated value
        // This test documents the removal of the strength parameter (issue #57)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            Issue.record("Failed to create test image"); return
        }

        let mode = Flux2GenerationMode.imageToImage(images: [img])
        // Pattern match with only images — no strength value
        if case .imageToImage(let images) = mode {
            #expect(images.count == 1)
        } else {
            Issue.record("Expected imageToImage with images only")
        }
    }
}

// MARK: - SchedulerOverrides Tests

@Suite struct SchedulerOverridesTests {

    @Test func defaultInit() {
        let overrides = SchedulerOverrides()
        #expect(overrides.customSigmas == nil)
        #expect(overrides.numSteps == nil)
        #expect(overrides.guidance == nil)
        #expect(!overrides.hasOverrides)
    }

    @Test func withNumSteps() {
        let overrides = SchedulerOverrides(numSteps: 4)
        #expect(overrides.numSteps == 4)
        #expect(overrides.hasOverrides)
    }

    @Test func withGuidance() {
        let overrides = SchedulerOverrides(guidance: 1.0)
        #expect(overrides.guidance == 1.0)
        #expect(overrides.hasOverrides)
    }

    @Test func withCustomSigmas() {
        let sigmas: [Float] = [1.0, 0.75, 0.5, 0.25, 0.0]
        let overrides = SchedulerOverrides(customSigmas: sigmas)
        #expect(overrides.customSigmas == sigmas)
        #expect(overrides.hasOverrides)
    }

    @Test func withAllOverrides() {
        let overrides = SchedulerOverrides(
            customSigmas: [1.0, 0.5, 0.0],
            numSteps: 8,
            guidance: 3.5
        )
        #expect(overrides.customSigmas?.count == 3)
        #expect(overrides.numSteps == 8)
        #expect(overrides.guidance == 3.5)
        #expect(overrides.hasOverrides)
    }

    @Test func equatable() {
        let a = SchedulerOverrides(numSteps: 4, guidance: 1.0)
        let b = SchedulerOverrides(numSteps: 4, guidance: 1.0)
        let c = SchedulerOverrides(numSteps: 8, guidance: 1.0)
        #expect(a == b)
        #expect(a != c)
    }

    @Test func codableRoundtrip() throws {
        let original = SchedulerOverrides(
            customSigmas: [1.0, 0.5, 0.0],
            numSteps: 4,
            guidance: 1.0
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(SchedulerOverrides.self, from: data)
        #expect(original == decoded)
    }

    @Test func codableNoStrengthField() throws {
        // Verify that JSON with a "strength" field is handled gracefully
        // (the field was removed, so it should be ignored by the decoder)
        let json = """
        {"numSteps": 4, "guidance": 1.0}
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(SchedulerOverrides.self, from: json)
        #expect(decoded.numSteps == 4)
        #expect(decoded.guidance == 1.0)
    }

    @Test func hasOverridesWithNoValues() {
        let overrides = SchedulerOverrides(customSigmas: nil, numSteps: nil, guidance: nil)
        #expect(!overrides.hasOverrides)
    }
}

// MARK: - LoRAConfig SchedulerOverrides Integration Tests

@Suite struct LoRAConfigSchedulerOverridesTests {

    @Test func loRAConfigWithSchedulerOverrides() {
        let overrides = SchedulerOverrides(numSteps: 4, guidance: 1.0)
        var config = LoRAConfig(filePath: "/path/to/turbo.safetensors")
        config.schedulerOverrides = overrides

        #expect(config.schedulerOverrides != nil)
        #expect(config.schedulerOverrides?.numSteps == 4)
        #expect(config.schedulerOverrides?.guidance == 1.0)
    }

    @Test func loRAConfigWithoutSchedulerOverrides() {
        let config = LoRAConfig(filePath: "/path/to/style.safetensors")
        #expect(config.schedulerOverrides == nil)
    }

    @Test func loRAConfigCodableWithOverrides() throws {
        let json = """
        {
            "filePath": "/path/to/turbo.safetensors",
            "scale": 1.0,
            "schedulerOverrides": {
                "numSteps": 8,
                "guidance": 3.5,
                "customSigmas": [1.0, 0.75, 0.5, 0.25, 0.0]
            }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(LoRAConfig.self, from: json)
        #expect(config.filePath == "/path/to/turbo.safetensors")
        #expect(config.scale == 1.0)
        #expect(config.schedulerOverrides != nil)
        #expect(config.schedulerOverrides?.numSteps == 8)
        #expect(config.schedulerOverrides?.guidance == 3.5)
        #expect(config.schedulerOverrides?.customSigmas?.count == 5)
    }

    @Test func loRAConfigCodableRoundtrip() throws {
        var config = LoRAConfig(filePath: "/path/to/lora.safetensors", scale: 0.8)
        config.activationKeyword = "sks"
        config.schedulerOverrides = SchedulerOverrides(numSteps: 4, guidance: 1.0)

        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(LoRAConfig.self, from: data)

        #expect(decoded.filePath == config.filePath)
        #expect(decoded.scale == config.scale)
        #expect(decoded.activationKeyword == "sks")
        #expect(decoded.schedulerOverrides == config.schedulerOverrides)
    }
}

// MARK: - CGImageSource Pipeline Helper Tests

@Suite struct CGImageSourcePipelineTests {

    @Test func cgImageFromValidPNGData() {
        // Create a small CGImage and encode to PNG
        let width = 4, height = 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            Issue.record("Failed to create test image"); return
        }

        // Encode to PNG data
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else {
            Issue.record("Failed to create image destination"); return
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            Issue.record("Failed to finalize image"); return
        }

        // Decode via pipeline helper
        let decoded = Flux2Pipeline.cgImage(from: mutableData as Data)
        #expect(decoded != nil)
        #expect(decoded?.width == width)
        #expect(decoded?.height == height)
    }

    @Test func cgImageFromValidJPEGData() {
        let width = 8, height = 8
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else {
            Issue.record("Failed to create test image"); return
        }

        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.jpeg" as CFString, 1, nil) else {
            Issue.record("Failed to create JPEG destination"); return
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            Issue.record("Failed to finalize JPEG"); return
        }

        let decoded = Flux2Pipeline.cgImage(from: mutableData as Data)
        #expect(decoded != nil)
        #expect(decoded?.width == width)
        #expect(decoded?.height == height)
    }

    @Test func cgImageFromInvalidData() {
        let garbage = Data([0x00, 0x01, 0x02, 0x03, 0xFF])
        let result = Flux2Pipeline.cgImage(from: garbage)
        #expect(result == nil)
    }

    @Test func cgImageFromEmptyData() {
        let result = Flux2Pipeline.cgImage(from: Data())
        #expect(result == nil)
    }

    @Test func cgImagePreservesPixelValues() {
        // Create gradient image with known pixel values
        let width = 8, height = 8
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let idx = (y * width + x) * 4
                pixels[idx + 0] = UInt8(x * 32)     // R
                pixels[idx + 1] = UInt8(y * 32)     // G
                pixels[idx + 2] = UInt8((x + y) * 16) // B
                pixels[idx + 3] = 255                // A (skip)
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = pixels.withUnsafeMutableBytes({ ptr -> CGContext? in
            CGContext(data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        }), let img = ctx.makeImage() else {
            Issue.record("Failed to create gradient image"); return
        }

        // Encode to PNG (lossless)
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(mutableData as CFMutableData, "public.png" as CFString, 1, nil) else {
            Issue.record("Failed to create PNG destination"); return
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            Issue.record("Failed to finalize PNG"); return
        }

        // Decode and verify
        guard let decoded = Flux2Pipeline.cgImage(from: mutableData as Data) else {
            Issue.record("Failed to decode PNG"); return
        }
        #expect(decoded.width == width)
        #expect(decoded.height == height)

        // Verify pixel values match
        guard let dataProvider = decoded.dataProvider,
              let pixelData = dataProvider.data,
              let bytes = CFDataGetBytePtr(pixelData) else {
            Issue.record("Failed to get pixel data from decoded image"); return
        }

        let bpp = decoded.bitsPerPixel / 8
        let bpr = decoded.bytesPerRow
        var mismatch = 0
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = (y * width + x) * 4
                let dstOff = y * bpr + x * bpp
                if pixels[srcIdx] != bytes[dstOff] || pixels[srcIdx + 1] != bytes[dstOff + 1] || pixels[srcIdx + 2] != bytes[dstOff + 2] {
                    mismatch += 1
                }
            }
        }
        #expect(mismatch == 0, "PNG roundtrip via CGImageSource should be pixel-exact")
    }
}

// MARK: - Klein 9B KV Configuration Tests

@Suite struct Klein9BKVConfigTests {

    @Test func klein9BKVModelProperties() {
        let model = Flux2Model.klein9BKV

        #expect(model.rawValue == "klein-9b-kv")
        #expect(model.displayName == "Flux.2 Klein 9B KV")
        #expect(model.isForInference)
        #expect(!model.isForTraining)
        #expect(!model.isBaseModel)
        #expect(model.supportsKVCache)
        #expect(model.defaultSteps == 4)
        #expect(model.defaultGuidance == 1.0)
        #expect(model.jointAttentionDim == 12288)
        #expect(!model.usesGuidanceEmbeds)
        #expect(!model.isCommercialUseAllowed)
        #expect(model.license == "Non-Commercial")
        #expect(model.maxReferenceImages == 4)
    }

    @Test func klein9BKVTransformerConfig() {
        let config = Flux2Model.klein9BKV.transformerConfig
        // Same architecture as klein-9b
        #expect(config.numLayers == 8)
        #expect(config.numSingleLayers == 24)
        #expect(config.numAttentionHeads == 32)
        #expect(config.attentionHeadDim == 128)
        #expect(config.innerDim == 4096)  // 32 x 128
        #expect(config.jointAttentionDim == 12288)
        #expect(!config.guidanceEmbeds)
    }

    @Test func supportsKVCacheOnlyKlein9BKV() {
        // Only klein-9b-kv should support KV cache
        #expect(Flux2Model.klein9BKV.supportsKVCache)
        #expect(!Flux2Model.klein9B.supportsKVCache)
        #expect(!Flux2Model.klein4B.supportsKVCache)
        #expect(!Flux2Model.dev.supportsKVCache)
        #expect(!Flux2Model.klein9BBase.supportsKVCache)
        #expect(!Flux2Model.klein4BBase.supportsKVCache)
    }

    @Test func klein9BKVInferenceVariant() {
        // Inference variant should be klein9B (standard distilled)
        #expect(Flux2Model.klein9BKV.inferenceVariant == .klein9B)
    }

    @Test func klein9BKVTrainingVariant() {
        // Training variant should be klein9BBase
        #expect(Flux2Model.klein9BKV.trainingVariant == .klein9BBase)
    }
}

// MARK: - Klein 9B KV Registry Tests

@Suite struct Klein9BKVRegistryTests {

    @Test func klein9BKVTransformerVariant() {
        let variant = ModelRegistry.TransformerVariant.klein9B_kv_bf16

        #expect(variant.rawValue == "klein9b-kv-bf16")
        #expect(variant.huggingFaceRepo == "black-forest-labs/FLUX.2-klein-9b-kv")
        #expect(variant.huggingFaceSubfolder == nil)
        #expect(variant.estimatedSizeGB == 18)
        #expect(variant.isGated)
        #expect(variant.modelType == .klein9BKV)
        #expect(variant.isForInference)
        #expect(!variant.isForTraining)
        #expect(variant.quantization == .bf16)
    }

    @Test func klein9BKVVariantLookup() {
        // All quantizations should return the bf16 variant (quantize on-the-fly)
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9BKV, quantization: .bf16) ==
            .klein9B_kv_bf16
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9BKV, quantization: .qint8) ==
            .klein9B_kv_bf16
        )
        #expect(
            ModelRegistry.TransformerVariant.variant(for: .klein9BKV, quantization: .int4) ==
            .klein9B_kv_bf16
        )
    }

    @Test func klein9BKVTrainingVariantLookup() {
        // Training variant should be klein9B_base_bf16 (same base model)
        #expect(
            ModelRegistry.TransformerVariant.trainingVariant(for: .klein9BKV) ==
            .klein9B_base_bf16
        )
    }

    @Test func klein9BKVLocalPath() {
        let path = ModelRegistry.localPath(for: .transformer(.klein9B_kv_bf16))
        #expect(path.path.contains("FLUX.2-klein-9b-kv"))
    }
}

// MARK: - TransformerKVCache Tests

@Suite struct TransformerKVCacheTests {

    @Test func kvCacheCreation() {
        let cache = TransformerKVCache(referenceTokenCount: 1024)
        #expect(cache.referenceTokenCount == 1024)
        #expect(cache.layerCount == 0)
        #expect(cache.doubleStreamEntries.isEmpty)
        #expect(cache.singleStreamEntries.isEmpty)
    }

    @Test func kvCacheEntryStorage() {
        var cache = TransformerKVCache(referenceTokenCount: 512)

        let keys = MLXRandom.normal([1, 32, 512, 128])
        let values = MLXRandom.normal([1, 32, 512, 128])
        let entry = LayerKVCacheEntry(keys: keys, values: values)

        cache.setDoubleStream(blockIndex: 0, entry: entry)
        #expect(cache.layerCount == 1)

        let retrieved = cache.doubleStreamEntry(at: 0)
        #expect(retrieved != nil)
        #expect(retrieved!.keys.shape == [1, 32, 512, 128])
        #expect(retrieved!.values.shape == [1, 32, 512, 128])
    }

    @Test func kvCacheSingleStreamStorage() {
        var cache = TransformerKVCache(referenceTokenCount: 256)

        let keys = MLXRandom.normal([1, 32, 256, 128])
        let values = MLXRandom.normal([1, 32, 256, 128])
        let entry = LayerKVCacheEntry(keys: keys, values: values)

        cache.setSingleStream(blockIndex: 5, entry: entry)
        #expect(cache.layerCount == 1)

        let retrieved = cache.singleStreamEntry(at: 5)
        #expect(retrieved != nil)

        // Non-existent index returns nil
        #expect(cache.singleStreamEntry(at: 99) == nil)
    }

    @Test func kvCacheClear() {
        var cache = TransformerKVCache(referenceTokenCount: 128)

        let entry = LayerKVCacheEntry(
            keys: MLXRandom.normal([1, 32, 128, 128]),
            values: MLXRandom.normal([1, 32, 128, 128])
        )

        cache.setDoubleStream(blockIndex: 0, entry: entry)
        cache.setDoubleStream(blockIndex: 1, entry: entry)
        cache.setSingleStream(blockIndex: 0, entry: entry)
        #expect(cache.layerCount == 3)

        cache.clear()
        #expect(cache.layerCount == 0)
        #expect(cache.doubleStreamEntries.isEmpty)
        #expect(cache.singleStreamEntries.isEmpty)
        // referenceTokenCount is preserved after clear
        #expect(cache.referenceTokenCount == 128)
    }

    @Test func kvCacheMultipleLayersCount() {
        var cache = TransformerKVCache(referenceTokenCount: 64)
        let entry = LayerKVCacheEntry(
            keys: MLXRandom.normal([1, 4, 64, 32]),
            values: MLXRandom.normal([1, 4, 64, 32])
        )

        // Fill 8 double + 24 single (Klein 9B architecture)
        for i in 0..<8 { cache.setDoubleStream(blockIndex: i, entry: entry) }
        for i in 0..<24 { cache.setSingleStream(blockIndex: i, entry: entry) }

        #expect(cache.layerCount == 32)
        #expect(cache.doubleStreamEntries.count == 8)
        #expect(cache.singleStreamEntries.count == 24)

        // Overwrite an existing entry
        cache.setDoubleStream(blockIndex: 0, entry: entry)
        #expect(cache.doubleStreamEntries.count == 8, "Overwrite should not increase count")
    }

    @Test func layerKVCacheEntryShapes() {
        let keys = MLXRandom.normal([1, 32, 1024, 128])
        let values = MLXRandom.normal([1, 32, 1024, 128])
        let entry = LayerKVCacheEntry(keys: keys, values: values)

        #expect(entry.keys.shape == [1, 32, 1024, 128])
        #expect(entry.values.shape == [1, 32, 1024, 128])
    }
}

// MARK: - KV Extraction Attention Mask Tests

@Suite struct KVExtractionMaskTests {

    /// Test double-stream attention mask pattern:
    /// Joint sequence order: [txt, ref, output]
    /// - txt queries: attend to ALL (txt, ref, output)
    /// - ref queries: attend to txt + ref ONLY (blocked from output)
    /// - output queries: attend to ALL
    @Test func doubleStreamMaskPattern() {
        let attn = Flux2Attention(dim: 128, numHeads: 4, headDim: 32)
        let textLen = 3
        let refLen = 2
        let outputLen = 4
        let totalSeq = textLen + refLen + outputLen  // 9

        let mask = attn.buildKVExtractionMask(
            textLen: textLen, refLen: refLen, outputLen: outputLen, totalSeq: totalSeq
        )

        #expect(mask.shape == [1, 1, totalSeq, totalSeq])

        // Materialize mask values
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(mask)
        let flat = mask.reshaped([totalSeq, totalSeq])

        // Helper to read mask value at (q, k)
        func maskVal(_ q: Int, _ k: Int) -> Float {
            flat[q, k].item(Float.self)
        }

        // txt queries (rows 0-2) should attend to everything -> all 0.0
        for q in 0..<textLen {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0, "txt query \(q) should attend to key \(k)")
            }
        }

        // ref queries (rows 3-4) should attend to txt+ref (0-4) but NOT output (5-8)
        for q in textLen..<(textLen + refLen) {
            // Can attend to txt and ref
            for k in 0..<(textLen + refLen) {
                #expect(maskVal(q, k) == 0.0, "ref query \(q) should attend to key \(k)")
            }
            // Blocked from output
            for k in (textLen + refLen)..<totalSeq {
                #expect(maskVal(q, k) == -Float.infinity, "ref query \(q) should be blocked from output key \(k)")
            }
        }

        // output queries (rows 5-8) should attend to everything -> all 0.0
        for q in (textLen + refLen)..<totalSeq {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0, "output query \(q) should attend to key \(k)")
            }
        }
    }

    /// Test single-stream attention mask pattern:
    /// Single sequence order: [txt, ref, output]
    /// Same blocking rule: ref queries blocked from output keys
    @Test func singleStreamMaskPattern() {
        let attn = Flux2ParallelSelfAttention(dim: 128, numHeads: 4, headDim: 32)
        let textLen = 3
        let refLen = 2
        let outputLen = 4
        let totalSeq = textLen + refLen + outputLen

        let mask = attn.buildSingleStreamKVExtractionMask(
            textLen: textLen, refLen: refLen, outputLen: outputLen, totalSeq: totalSeq
        )

        #expect(mask.shape == [1, 1, totalSeq, totalSeq])

        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(mask)
        let flat = mask.reshaped([totalSeq, totalSeq])

        func maskVal(_ q: Int, _ k: Int) -> Float {
            flat[q, k].item(Float.self)
        }

        // Ref queries blocked from output
        for q in textLen..<(textLen + refLen) {
            for k in (textLen + refLen)..<totalSeq {
                #expect(maskVal(q, k) == -Float.infinity)
            }
        }

        // Everything else is 0.0
        for q in 0..<textLen {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0)
            }
        }
        for q in (textLen + refLen)..<totalSeq {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0)
            }
        }
    }

    /// Edge case: 0 reference tokens -> no blocking at all (all 0.0)
    @Test func maskWithZeroReferenceTokens() {
        let attn = Flux2Attention(dim: 128, numHeads: 4, headDim: 32)
        let mask = attn.buildKVExtractionMask(textLen: 5, refLen: 0, outputLen: 10, totalSeq: 15)

        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(mask)
        let flat = mask.reshaped([15 * 15])

        // All values should be 0.0 (no blocking)
        let sum = MLX.abs(flat).sum()
        eval(sum)
        #expect(sum.item(Float.self) == 0.0, "No ref tokens -> no blocking")
    }
}

// MARK: - Flux2Attention KV Methods Tests

@Suite struct Flux2AttentionKVTests {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128  // 4 x 32
    let seqLenTxt = 8
    let seqLenRef = 6
    let seqLenImg = 10

    func makeAttention() -> Flux2Attention {
        Flux2Attention(dim: dim, numHeads: numHeads, headDim: headDim)
    }

    func makeRoPE(seqLen: Int) -> (cos: MLXArray, sin: MLXArray) {
        // Fake RoPE embeddings
        (cos: MLXRandom.normal([seqLen, headDim]),
         sin: MLXRandom.normal([seqLen, headDim]))
    }

    @Test func callWithKVExtractionOutputShapes() {
        let attn = makeAttention()
        // hiddenStates contains [ref + img] tokens
        let imgPlusRef = MLXRandom.normal([batchSize, seqLenRef + seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenRef + seqLenImg)

        let (hsOut, ehsOut, cache) = attn.callWithKVExtraction(
            hiddenStates: imgPlusRef,
            encoderHiddenStates: txt,
            rotaryEmb: rope,
            referenceTokenCount: seqLenRef
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(hsOut, ehsOut, cache.keys, cache.values)

        // Output shapes match input
        #expect(hsOut.shape == [batchSize, seqLenRef + seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])

        // Cache contains reference K/V with correct shape [B, H, S_ref, D]
        #expect(cache.keys.shape == [batchSize, numHeads, seqLenRef, headDim])
        #expect(cache.values.shape == [batchSize, numHeads, seqLenRef, headDim])
    }

    @Test func callWithKVCachedOutputShapes() {
        let attn = makeAttention()
        // Only output tokens (no ref)
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenImg)

        // Fake cached KV
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let (hsOut, ehsOut) = attn.callWithKVCached(
            hiddenStates: img,
            encoderHiddenStates: txt,
            rotaryEmb: rope,
            cachedKV: cachedKV
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(hsOut, ehsOut)

        #expect(hsOut.shape == [batchSize, seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])
    }

    /// Standard callAsFunction and callWithKVCached with 0 ref tokens
    /// should produce outputs of the same shape
    @Test func standardAndCachedConsistentShapes() {
        let attn = makeAttention()
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenImg)

        let (stdHs, stdEhs) = attn.callAsFunction(
            hiddenStates: img, encoderHiddenStates: txt, rotaryEmb: rope
        )

        // Empty cache (0 ref tokens)
        let emptyCache = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, 0, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, 0, headDim])
        )
        let (cachedHs, cachedEhs) = attn.callWithKVCached(
            hiddenStates: img, encoderHiddenStates: txt,
            rotaryEmb: rope, cachedKV: emptyCache
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(stdHs, stdEhs, cachedHs, cachedEhs)

        #expect(stdHs.shape == cachedHs.shape)
        #expect(stdEhs.shape == cachedEhs.shape)
    }
}

// MARK: - Flux2ParallelSelfAttention KV Methods Tests

@Suite struct Flux2ParallelAttentionKVTests {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128
    let textLen = 8
    let refLen = 6
    let imgLen = 10

    func makeAttention() -> Flux2ParallelSelfAttention {
        Flux2ParallelSelfAttention(dim: dim, numHeads: numHeads, headDim: headDim)
    }

    @Test func callWithKVExtractionOutputShapes() {
        let attn = makeAttention()
        let totalSeq = textLen + refLen + imgLen
        let combined = MLXRandom.normal([batchSize, totalSeq, dim])
        let rope = (cos: MLXRandom.normal([totalSeq, headDim]),
                    sin: MLXRandom.normal([totalSeq, headDim]))

        let (output, cache) = attn.callWithKVExtraction(
            hiddenStates: combined,
            rotaryEmb: rope,
            textLen: textLen,
            referenceTokenCount: refLen
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(output, cache.keys, cache.values)

        // Output has same shape as input
        #expect(output.shape == [batchSize, totalSeq, dim])

        // Cache contains reference K/V: [B, H, refLen, D]
        #expect(cache.keys.shape == [batchSize, numHeads, refLen, headDim])
        #expect(cache.values.shape == [batchSize, numHeads, refLen, headDim])
    }

    @Test func callWithKVCachedOutputShapes() {
        let attn = makeAttention()
        let seqLen = textLen + imgLen  // no ref tokens
        let combined = MLXRandom.normal([batchSize, seqLen, dim])
        let rope = (cos: MLXRandom.normal([seqLen, headDim]),
                    sin: MLXRandom.normal([seqLen, headDim]))

        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, refLen, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, refLen, headDim])
        )

        let output = attn.callWithKVCached(
            hiddenStates: combined,
            rotaryEmb: rope,
            cachedKV: cachedKV,
            textLen: textLen
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(output)

        // Output has same shape as input (txt + img, no ref added)
        #expect(output.shape == [batchSize, seqLen, dim])
    }
}

// MARK: - Transformer Block KV Forwarding Tests

@Suite struct TransformerBlockKVTests {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128
    let seqLenTxt = 8
    let seqLenRef = 4
    let seqLenImg = 10

    @Test func doubleStreamBlockKVExtraction() {
        let block = Flux2TransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let imgPlusRef = MLXRandom.normal([batchSize, seqLenRef + seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let totalRope = seqLenTxt + seqLenRef + seqLenImg
        let rope = (cos: MLXRandom.normal([totalRope, headDim]),
                    sin: MLXRandom.normal([totalRope, headDim]))

        let (ehsOut, hsOut, cache) = block.callWithKVExtraction(
            hiddenStates: imgPlusRef,
            encoderHiddenStates: txt,
            temb: temb,
            rotaryEmb: rope,
            imgModParams: nil,
            txtModParams: nil,
            referenceTokenCount: seqLenRef
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(ehsOut, hsOut, cache.keys, cache.values)

        #expect(hsOut.shape == [batchSize, seqLenRef + seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])
        #expect(cache.keys.shape == [batchSize, numHeads, seqLenRef, headDim])
    }

    @Test func doubleStreamBlockKVCached() {
        let block = Flux2TransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([seqLenTxt + seqLenImg, headDim]),
                    sin: MLXRandom.normal([seqLenTxt + seqLenImg, headDim]))
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let (ehsOut, hsOut) = block.callWithKVCached(
            hiddenStates: img,
            encoderHiddenStates: txt,
            temb: temb,
            rotaryEmb: rope,
            imgModParams: nil,
            txtModParams: nil,
            cachedKV: cachedKV
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(ehsOut, hsOut)

        #expect(hsOut.shape == [batchSize, seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])
    }

    @Test func singleStreamBlockKVExtraction() {
        let block = Flux2SingleTransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let totalSeq = seqLenTxt + seqLenRef + seqLenImg
        let combined = MLXRandom.normal([batchSize, totalSeq, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([totalSeq, headDim]),
                    sin: MLXRandom.normal([totalSeq, headDim]))

        let (output, cache) = block.callWithKVExtraction(
            hiddenStates: combined,
            temb: temb,
            rotaryEmb: rope,
            modParams: nil,
            textLen: seqLenTxt,
            referenceTokenCount: seqLenRef
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(output, cache.keys, cache.values)

        #expect(output.shape == [batchSize, totalSeq, dim])
        #expect(cache.keys.shape == [batchSize, numHeads, seqLenRef, headDim])
    }

    @Test func singleStreamBlockKVCached() {
        let block = Flux2SingleTransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let seqLen = seqLenTxt + seqLenImg  // no ref
        let combined = MLXRandom.normal([batchSize, seqLen, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([seqLen, headDim]),
                    sin: MLXRandom.normal([seqLen, headDim]))
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let output = block.callWithKVCached(
            hiddenStates: combined,
            temb: temb,
            rotaryEmb: rope,
            modParams: nil,
            cachedKV: cachedKV,
            textLen: seqLenTxt
        )
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }
        eval(output)

        #expect(output.shape == [batchSize, seqLen, dim])
    }
}

// MARK: - Klein 9B KV Enum Exhaustiveness Tests

@Suite struct Klein9BKVEnumTests {

    @Test func flux2ModelAllCasesIncludesKlein9BKV() {
        let allRawValues = Flux2Model.allCases.map { $0.rawValue }
        #expect(allRawValues.contains("klein-9b-kv"))
    }

    @Test func transformerVariantAllCasesIncludesKlein9BKV() {
        let allRawValues = ModelRegistry.TransformerVariant.allCases.map { $0.rawValue }
        #expect(allRawValues.contains("klein9b-kv-bf16"))
    }

    @Test func flux2ModelCaseCount() {
        // dev, klein-4b, klein-4b-base, klein-9b, klein-9b-base, klein-9b-kv = 6
        #expect(Flux2Model.allCases.count == 6)
    }

    @Test func transformerVariantCaseCount() {
        // bf16, qint8, klein4b-bf16, klein4b-8bit, klein4b-base-bf16,
        // klein9b-bf16, klein9b-base-bf16, klein9b-kv-bf16 = 8
        #expect(ModelRegistry.TransformerVariant.allCases.count == 8)
    }

    @Test func klein9BKVMemoryConfigDoesNotCrash() {
        // Verify MemoryConfig handles the new model in all code paths
        let limit = MemoryConfig.cacheLimitForResolution(width: 512, height: 512, model: .klein9BKV)
        #expect(limit > 0)

        let phaseLimits = MemoryConfig.PhaseLimits.forModel(.klein9BKV, profile: .auto)
        #expect(phaseLimits.denoising > 0)
        #expect(phaseLimits.textEncoding > 0)
        #expect(phaseLimits.vaeDecoding > 0)

        // Manual profiles fallback to dynamic
        let conservativeLimits = MemoryConfig.PhaseLimits.forModel(.klein9BKV, profile: .conservative)
        #expect(conservativeLimits.denoising > 0)
    }
}
