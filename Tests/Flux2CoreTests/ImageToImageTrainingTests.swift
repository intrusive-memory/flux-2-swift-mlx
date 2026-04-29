// ImageToImageTrainingTests.swift - Unit tests for Image-to-Image LoRA training
// Copyright 2025 Vincent Gourbin

import Testing
import Foundation
@testable import Flux2Core

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

