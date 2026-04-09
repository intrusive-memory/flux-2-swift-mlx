// ImageToImageTrainingTests.swift - Unit tests for Image-to-Image LoRA training
// Copyright 2025 Vincent Gourbin

import Testing
import Foundation
@testable import Flux2Core
import MLX

// MARK: - LoRATrainingConfig I2I Tests

@Suite struct ImageToImageConfigTests {

    let tmpDataset = URL(fileURLWithPath: "/tmp/test-dataset")
    let tmpOutput = URL(fileURLWithPath: "/tmp/test-output")

    // MARK: - controlPath / controlDropout / isImageToImage

    @Test func configDefaultsToTextToImage() {
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            outputPath: tmpOutput
        )

        #expect(config.controlPath == nil)
        #expect(config.controlDropout == 0.0)
        #expect(!config.isImageToImage)
    }

    @Test func configWithControlPathIsI2I() {
        let controlURL = URL(fileURLWithPath: "/tmp/controls")
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            controlPath: controlURL,
            controlDropout: 0.3,
            outputPath: tmpOutput
        )

        #expect(config.controlPath == controlURL)
        #expect(config.controlDropout == 0.3)
        #expect(config.isImageToImage)
    }

    @Test func configNilControlPathIsT2I() {
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            controlPath: nil,
            controlDropout: 0.5,
            outputPath: tmpOutput
        )

        #expect(!config.isImageToImage)
    }

    @Test func configControlDropoutDefaultsToZero() {
        let config = LoRATrainingConfig(
            datasetPath: tmpDataset,
            controlPath: URL(fileURLWithPath: "/tmp/controls"),
            outputPath: tmpOutput
        )

        #expect(config.controlDropout == 0.0)
    }

    // MARK: - ValidationPromptConfig.referenceImage

    @Test func validationPromptDefaultNoReferenceImage() {
        let prompt = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "a cat sitting",
            is512: true
        )

        #expect(prompt.referenceImage == nil)
    }

    @Test func validationPromptWithReferenceImage() {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let prompt = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "remove the hat",
            is512: true,
            referenceImage: refURL
        )

        #expect(prompt.referenceImage == refURL)
    }

    // MARK: - Codable round-trip (referenceImage)

    @Test func validationPromptCodableRoundTrip() throws {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let original = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "edit instruction",
            is512: true,
            is1024: false,
            applyTrigger: true,
            seed: 42,
            referenceImage: refURL
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(LoRATrainingConfig.ValidationPromptConfig.self, from: data)

        #expect(decoded.prompt == "edit instruction")
        #expect(decoded.is512 == true)
        #expect(decoded.is1024 == false)
        #expect(decoded.applyTrigger == true)
        #expect(decoded.seed == 42)
        #expect(decoded.referenceImage == refURL)
    }

    @Test func validationPromptCodableNilReferenceImage() throws {
        let original = LoRATrainingConfig.ValidationPromptConfig(
            prompt: "a landscape",
            is512: true
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(
            LoRATrainingConfig.ValidationPromptConfig.self, from: data
        )

        #expect(decoded.referenceImage == nil)
    }

    // MARK: - Presets remain T2I

    @Test func presetsHaveNoControlPath() {
        let minimal = LoRATrainingConfig.minimal8GB(
            datasetPath: tmpDataset, outputPath: tmpOutput
        )
        #expect(minimal.controlPath == nil)
        #expect(!minimal.isImageToImage)

        let balanced = LoRATrainingConfig.balanced16GB(
            datasetPath: tmpDataset, outputPath: tmpOutput
        )
        #expect(balanced.controlPath == nil)
        #expect(!balanced.isImageToImage)

        let quality = LoRATrainingConfig.quality32GB(
            datasetPath: tmpDataset, outputPath: tmpOutput
        )
        #expect(quality.controlPath == nil)
        #expect(!quality.isImageToImage)
    }
}

// MARK: - CachedLatentEntry I2I Tests

@Suite struct CachedLatentEntryI2ITests {

    @Test func cachedLatentEntryDefaultNoControl() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let entry = CachedLatentEntry(
            filename: "test.png",
            latent: MLXArray.zeros([1, 32, 64, 64]),
            width: 512,
            height: 512
        )

        #expect(entry.controlLatent == nil)
        #expect(entry.filename == "test.png")
        #expect(entry.width == 512)
        #expect(entry.height == 512)
    }

    @Test func cachedLatentEntryWithControlLatent() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let targetLatent = MLXArray.zeros([1, 32, 64, 64])
        let controlLatent = MLXArray.ones([1, 32, 64, 64])

        let entry = CachedLatentEntry(
            filename: "edit_001.png",
            latent: targetLatent,
            width: 512,
            height: 512,
            controlLatent: controlLatent
        )

        #expect(entry.controlLatent != nil)
        #expect(entry.controlLatent!.shape == [1, 32, 64, 64])
    }

    @Test func cachedLatentEntryControlLatentMatchesTargetShape() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let entry = CachedLatentEntry(
            filename: "pair.png",
            latent: target,
            width: w * 8,
            height: h * 8,
            controlLatent: control
        )

        // Control and target should have same spatial dimensions
        #expect(entry.latent.shape == entry.controlLatent!.shape)
    }
}

// MARK: - SimpleLoRAConfig I2I Tests

@Suite struct SimpleLoRAConfigI2ITests {

    @Test func simpleLoRAConfigDefaultControlDropout() {
        let config = SimpleLoRAConfig(outputDir: URL(fileURLWithPath: "/tmp"))

        #expect(config.controlDropout == 0.0)
    }

    @Test func simpleLoRAConfigControlDropoutSetting() {
        var config = SimpleLoRAConfig(outputDir: URL(fileURLWithPath: "/tmp"))
        config.controlDropout = 0.3

        #expect(config.controlDropout == 0.3)
    }

    @Test func simpleLoRAConfigValidationPromptReferenceImage() {
        let refURL = URL(fileURLWithPath: "/tmp/ref.png")
        let prompt = SimpleLoRAConfig.ValidationPromptConfig(
            prompt: "remove background",
            is512: true,
            applyTrigger: false,
            referenceImage: refURL
        )

        #expect(prompt.referenceImage == refURL)
    }

    @Test func simpleLoRAConfigValidationPromptNoReferenceImage() {
        let prompt = SimpleLoRAConfig.ValidationPromptConfig(
            prompt: "a cat",
            is512: true,
            applyTrigger: true,
            seed: 42
        )

        #expect(prompt.referenceImage == nil)
    }
}

// MARK: - Position ID Tests for I2I

@Suite struct PositionIDI2ITests {

    @Test func referenceImagePositionIDsGeneration() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let height = 512
        let width = 512
        let latentH = height / 8  // 64
        let latentW = width / 8   // 64

        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH,
            latentWidth: latentW,
            imageIndex: 0
        )

        // Position count: latentH * latentW = 4096 (patchification is separate)
        let expectedPatches = latentH * latentW
        #expect(refImgIds.shape[0] == expectedPatches)
        #expect(refImgIds.shape[1] == 4)  // [T, H, W, L]
    }

    @Test func referenceImagePositionIDsTCoordinate() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let latentH = 64
        let latentW = 64

        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH,
            latentWidth: latentW,
            imageIndex: 0
        )

        eval(refImgIds)

        // T coordinate for first reference image should be 10 (T=10 + imageIndex*10)
        let firstT = refImgIds[0, 0].item(Int32.self)
        #expect(firstT == 10, "First reference image should have T=10")
    }

    @Test func referenceImagePositionIDsSecondImage() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let latentH = 64
        let latentW = 64

        let refIds0 = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH, latentWidth: latentW, imageIndex: 0
        )
        let refIds1 = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH, latentWidth: latentW, imageIndex: 1
        )

        eval(refIds0, refIds1)

        let t0 = refIds0[0, 0].item(Int32.self)
        let t1 = refIds1[0, 0].item(Int32.self)

        // Different images should have different T coordinates
        #expect(t0 != t1)
    }

    @Test func concatenatedPositionIDs() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let height = 512
        let width = 512

        let imgIds = LatentUtils.generateImagePositionIDs(height: height, width: width)
        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: height / 8,
            latentWidth: width / 8,
            imageIndex: 0
        )

        // Concatenate along sequence dimension
        let combined = concatenated([imgIds, refImgIds], axis: 0)

        let expectedTotal = imgIds.shape[0] + refImgIds.shape[0]
        #expect(combined.shape[0] == expectedTotal)
        #expect(combined.shape[1] == 4)
    }
}

// MARK: - Latent Packing I2I Tests

@Suite struct LatentPackingI2ITests {

    @Test func packedControlLatentSameShapeAsTarget() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let packedTarget = LatentUtils.packLatents(target, patchSize: 2)
        let packedControl = LatentUtils.packLatents(control, patchSize: 2)

        #expect(packedTarget.shape == packedControl.shape)
    }

    @Test func concatenatedLatentSequence() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let packedTarget = LatentUtils.packLatents(target, patchSize: 2)
        let packedControl = LatentUtils.packLatents(control, patchSize: 2)

        // Concatenate along sequence dimension
        let combined = concatenated([packedTarget, packedControl], axis: 1)

        let outputSeqLen = packedTarget.shape[1]
        let totalSeqLen = combined.shape[1]

        #expect(totalSeqLen == 2 * outputSeqLen)
        #expect(combined.shape[0] == 1)  // batch
        #expect(combined.shape[2] == 128)  // channels
    }

    @Test func outputSlicingFromCombinedLatent() {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI { return }

        let seqLen = 1024
        let channels = 128

        // Simulate model output with combined sequence
        let modelOutput = MLXRandom.normal([1, 2 * seqLen, channels])

        // Slice output portion only
        let outputPortion = modelOutput[0..., 0..<seqLen, 0...]

        eval(outputPortion)
        #expect(outputPortion.shape == [1, seqLen, channels])
    }
}
