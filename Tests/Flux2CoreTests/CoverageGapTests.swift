// CoverageGapTests.swift - WU-5 S8: Coverage gap tests for Flux2Core
// Copyright 2025

import Testing
import Foundation
@testable import Flux2Core
import MLX

@Suite("CoverageGaps")
struct CoverageGapTests {

    // Test 1: Flux2 transformer config defaults are positive integers
    @Test(.timeLimit(.minutes(1)))
    func flux2TransformerConfigDefaults() {
        let config = Flux2TransformerConfig.flux2Dev
        #expect(config.numAttentionHeads > 0)
        #expect(config.numLayers > 0)
        #expect(config.innerDim > 0)
        #expect(config.attentionHeadDim > 0)
        #expect(config.numSingleLayers > 0)
    }

    // Test 2: Custom init round-trips all fields correctly
    @Test(.timeLimit(.minutes(1)))
    func flux2TransformerConfigCustomInit() {
        let config = Flux2TransformerConfig(
            patchSize: 2,
            inChannels: 64,
            outChannels: 64,
            numLayers: 4,
            numSingleLayers: 16,
            attentionHeadDim: 64,
            numAttentionHeads: 16,
            jointAttentionDim: 4096,
            pooledProjectionDim: 512,
            guidanceEmbeds: false,
            axesDimsRope: [16, 16, 16, 16],
            ropeTheta: 1000.0,
            mlpRatio: 2.0,
            activationFunction: "gelu"
        )
        #expect(config.patchSize == 2)
        #expect(config.inChannels == 64)
        #expect(config.outChannels == 64)
        #expect(config.numLayers == 4)
        #expect(config.numSingleLayers == 16)
        #expect(config.attentionHeadDim == 64)
        #expect(config.numAttentionHeads == 16)
        #expect(config.jointAttentionDim == 4096)
        #expect(config.pooledProjectionDim == 512)
        #expect(config.guidanceEmbeds == false)
        #expect(config.axesDimsRope == [16, 16, 16, 16])
        #expect(config.ropeTheta == 1000.0)
        #expect(config.mlpRatio == 2.0)
        #expect(config.activationFunction == "gelu")
        // innerDim is computed: 16 * 64 = 1024
        #expect(config.innerDim == 1024)
    }

    // Test 3: VAE config scaling factor and channel counts are positive
    @Test(.timeLimit(.minutes(1)))
    func vaeConfigScalingFactors() {
        let config = VAEConfig.flux2Dev
        #expect(config.scalingFactor > 0)
        #expect(config.inChannels > 0)
        #expect(config.outChannels > 0)
        #expect(config.latentChannels > 0)
        #expect(config.normNumGroups > 0)
        #expect(!config.blockOutChannels.isEmpty)
    }

    // Test 4: Quantization presets are distinct in at least one field
    @Test(.timeLimit(.minutes(1)))
    func quantizationPresetsAreDistinct() {
        let highQuality = Flux2QuantizationConfig.highQuality
        let balanced = Flux2QuantizationConfig.balanced
        let ultraMinimal = Flux2QuantizationConfig.ultraMinimal

        // highQuality uses bf16 transformer, ultraMinimal uses int4
        #expect(highQuality.transformer != ultraMinimal.transformer)
        // balanced differs from highQuality in text encoder
        #expect(highQuality.textEncoder != balanced.textEncoder)
        // ultraMinimal has lower estimated memory than highQuality
        #expect(ultraMinimal.estimatedTotalMemoryGB < highQuality.estimatedTotalMemoryGB)
    }

    // Test 5: Model registry maps all TransformerVariant cases with non-empty strings
    @Test(.timeLimit(.minutes(1)))
    func modelRegistryMapsAllCases() {
        for variant in ModelRegistry.TransformerVariant.allCases {
            #expect(!variant.huggingFaceRepo.isEmpty,
                    "huggingFaceRepo is empty for \(variant.rawValue)")
            #expect(!variant.rawValue.isEmpty,
                    "rawValue is empty for variant")
        }
    }

    // Test 6: FlowMatchEulerScheduler timestep count matches requested steps
    @Test(.timeLimit(.minutes(1)))
    func flowMatchSchedulerTimestepCount() {
        let scheduler = FlowMatchEulerScheduler()
        let n = 20
        scheduler.setTimesteps(numInferenceSteps: n)
        // timesteps.count == sigmas.count (includes terminal sigma 0)
        // effective steps = sigmas.count - 1 == n
        #expect(scheduler.timesteps.count == n + 1)
    }

    // Test 7: Sigmas decrease monotonically (first > last)
    @Test(.timeLimit(.minutes(1)))
    func flowMatchSchedulerFirstLastSigma() {
        let scheduler = FlowMatchEulerScheduler()
        scheduler.setTimesteps(numInferenceSteps: 10)
        let sigmas = scheduler.sigmas
        #expect(!sigmas.isEmpty)
        #expect(sigmas.first! > sigmas.last!)
    }

    // Test 8: packPatchifiedToSequence / unpackSequenceToPatchified round-trip at 512x512
    @Test(.timeLimit(.minutes(1)))
    func latentUtilsRoundTrip512() {
        let height = 512
        let width = 512
        // patchified shape: [1, 128, H/16, W/16] = [1, 128, 32, 32]
        let pH = height / 16
        let pW = width / 16
        let patchified = MLXArray.zeros([1, 128, pH, pW])
        let sequence = LatentUtils.packPatchifiedToSequence(patchified)
        let restored = LatentUtils.unpackSequenceToPatchified(sequence, height: height, width: width)
        // shape should be restored
        #expect(restored.shape == patchified.shape)
    }

    // Test 9: packPatchifiedToSequence / unpackSequenceToPatchified round-trip at 1024x1024
    @Test(.timeLimit(.minutes(1)))
    func latentUtilsRoundTrip1024() {
        let height = 1024
        let width = 1024
        let pH = height / 16
        let pW = width / 16
        let patchified = MLXArray.zeros([1, 128, pH, pW])
        let sequence = LatentUtils.packPatchifiedToSequence(patchified)
        let restored = LatentUtils.unpackSequenceToPatchified(sequence, height: height, width: width)
        #expect(restored.shape == patchified.shape)
    }

    // Test 10: validateDimensions rounds up non-multiples-of-16 to valid dimensions
    @Test(.timeLimit(.minutes(1)))
    func latentUtilsRejectsInvalidDimensions() {
        // validateDimensions rounds non-multiples to the next valid size
        let (h, w) = LatentUtils.validateDimensions(height: 500, width: 500)
        // Must be divisible by 16 (8 * patchSize=2)
        #expect(h % 16 == 0)
        #expect(w % 16 == 0)
        #expect(h >= 500)
        #expect(w >= 500)

        // A valid multiple should remain unchanged
        let (h2, w2) = LatentUtils.validateDimensions(height: 512, width: 512)
        #expect(h2 == 512)
        #expect(w2 == 512)
    }

    // Test 11: generateImagePositionIDs is deterministic for same resolution
    @Test(.timeLimit(.minutes(1)))
    func ropePositionIdsAreDeterministic() {
        let ids1 = LatentUtils.generateImagePositionIDs(height: 512, width: 512)
        let ids2 = LatentUtils.generateImagePositionIDs(height: 512, width: 512)
        // Same shape
        #expect(ids1.shape == ids2.shape)
        // Values are equal (deterministic, no randomness)
        let diff = MLX.abs(ids1 - ids2)
        let maxDiff = diff.max().item(Int.self)
        #expect(maxDiff == 0)
    }

    // Test 12: Flux2MemoryManager - Klein 4B bf16 threshold is at least 8 GB
    @Test(.timeLimit(.minutes(1)))
    func memoryManagerKlein4BThreshold() {
        // Klein 4B bf16 requires 16 GB per Training.recommendedMemoryGB
        let threshold = Training.recommendedMemoryGB(for: .klein4B, quantization: .bf16)
        #expect(threshold >= 8, "Klein 4B bf16 threshold should be at least 8 GB, got \(threshold)")
    }

    // Test 13: Largest model threshold is larger than smallest model threshold
    @Test(.timeLimit(.minutes(1)))
    func memoryManagerKlein9BThreshold() {
        let smallest = Training.recommendedMemoryGB(for: .klein4B, quantization: .nf4)
        let largest = Training.recommendedMemoryGB(for: .dev, quantization: .bf16)
        #expect(largest > smallest,
                "dev bf16 (\(largest)GB) should require more memory than klein4B nf4 (\(smallest)GB)")
    }

    // Test 14: All Flux2Error cases have non-nil, non-empty localizedDescription
    @Test(.timeLimit(.minutes(1)))
    func allFlux2ErrorCasesHaveContext() {
        let cases: [Flux2Error] = [
            .modelNotLoaded("transformer"),
            .invalidConfiguration("bad config"),
            .insufficientMemory(required: 16, available: 8),
            .weightLoadingFailed("missing file"),
            .imageProcessingFailed("decode failed"),
            .generationFailed("timeout"),
            .generationCancelled
        ]
        for error in cases {
            let desc = error.errorDescription
            #expect(desc != nil, "errorDescription is nil for \(error)")
            if let desc = desc {
                #expect(!desc.isEmpty, "errorDescription is empty for \(error)")
            }
        }
    }

    // Test 15: LoRAConfig conforms to Codable — JSON round-trip preserves equality
    @Test(.timeLimit(.minutes(1)))
    func loraConfigYamlRoundTrip() throws {
        let original = LoRAConfig(
            filePath: "/tmp/test.safetensors",
            scale: 0.8,
            activationKeyword: "sks",
            schedulerOverrides: nil
        )
        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(LoRAConfig.self, from: data)

        #expect(decoded.filePath == original.filePath)
        #expect(decoded.scale == original.scale)
        #expect(decoded.activationKeyword == original.activationKeyword)
        #expect(decoded.effectiveScale == original.effectiveScale)
    }

    // Test 16: TrainingState initializes with zero step and infinite best loss
    @Test(.timeLimit(.minutes(1)))
    func trainingStateInitializesToZero() {
        let state = TrainingState(
            currentStep: 0,
            totalSteps: 100,
            rngSeed: 42,
            configHash: "abc123",
            modelType: "klein4B",
            loraRank: 16,
            loraAlpha: 16.0
        )
        #expect(state.currentStep == 0)
        #expect(state.recentLosses.isEmpty)
        #expect(state.bestLoss == Float.infinity)
        #expect(state.checkpointSteps.isEmpty)
    }

    // Test 17: TrainingController pause sets shouldPause, resume clears it
    @Test(.timeLimit(.minutes(1)))
    func trainingControllerPauseResume() {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tc-test-\(UInt64.random(in: 0..<UInt64.max))")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let controller = TrainingController(outputDirectory: tmpDir)
        // Initial state: should not pause
        #expect(!controller.shouldPause())

        // Request pause
        controller.requestPause()
        #expect(controller.shouldPause())

        // Resume clears pause flag and file
        controller.resume()
        #expect(!controller.shouldPause())
    }

    // Test 18: TrainingController stop sets shouldStop
    @Test(.timeLimit(.minutes(1)))
    func trainingControllerStop() {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tc-stop-\(UInt64.random(in: 0..<UInt64.max))")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let controller = TrainingController(outputDirectory: tmpDir)
        #expect(!controller.shouldStop())

        controller.requestStop()
        #expect(controller.shouldStop())
    }

    // Test 19: AspectRatioBucketManager assigns known image sizes to non-nil buckets
    @Test(.timeLimit(.minutes(1)))
    func aspectRatioBucketAssignment() {
        let manager = AspectRatioBucketManager(resolutions: [512, 768, 1024])
        #expect(!manager.buckets.isEmpty)

        // Assign known sizes and verify they get a bucket
        let testCases: [(width: Int, height: Int)] = [
            (512, 512),
            (1024, 768),
            (768, 1024),
            (1920, 1080)
        ]

        for tc in testCases {
            let bucket = manager.findBestBucket(width: tc.width, height: tc.height)
            #expect(bucket.width > 0, "bucket width should be positive for \(tc.width)x\(tc.height)")
            #expect(bucket.height > 0, "bucket height should be positive for \(tc.width)x\(tc.height)")
        }
    }
}
