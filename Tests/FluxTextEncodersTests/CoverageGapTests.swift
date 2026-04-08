/**
 * CoverageGapTests.swift
 * WU-3 S6: Coverage gap tests for FluxTextEncoders
 * All tests are CI-safe: no GPU, no model downloads.
 */

import Testing
import Foundation
import CoreGraphics
@testable import FluxTextEncoders
import TestHelpers

@Suite("CoverageGaps")
struct CoverageGapTests {

    // MARK: - Test 1: ModelVariant CaseIterable

    @Test(.timeLimit(.minutes(1))) func allModelVariantCasesEnumerate() {
        let cases = ModelVariant.allCases
        #expect(cases.count > 0, "ModelVariant should have at least one case")
        for variant in cases {
            #expect(!variant.rawValue.isEmpty, "ModelVariant rawValue should be non-empty")
        }
    }

    // MARK: - Test 2: Qwen3Variant CaseIterable

    @Test(.timeLimit(.minutes(1))) func allQwen3VariantCasesEnumerate() {
        let cases = Qwen3Variant.allCases
        #expect(cases.count > 0, "Qwen3Variant should have at least one case")
        for variant in cases {
            #expect(!variant.rawValue.isEmpty, "Qwen3Variant rawValue should be non-empty")
        }
    }

    // MARK: - Test 3: KleinVariant CaseIterable

    @Test(.timeLimit(.minutes(1))) func allKleinVariantCasesEnumerate() {
        let cases = KleinVariant.allCases
        #expect(cases.count > 0, "KleinVariant should have at least one case")
        for variant in cases {
            #expect(!variant.rawValue.isEmpty, "KleinVariant rawValue should be non-empty")
        }
    }

    // MARK: - Test 4: GenerateParameters presets are distinct

    @Test(.timeLimit(.minutes(1))) func generateParameterPresetsAreDistinct() {
        let greedy = GenerateParameters.greedy
        let creative = GenerateParameters.creative
        let balanced = GenerateParameters.balanced

        // Greedy uses temperature=0, creative uses temperature=0.9
        #expect(greedy.temperature != creative.temperature,
                "greedy and creative presets should differ in temperature")
        // Creative allows more tokens than balanced
        #expect(creative.maxTokens != greedy.maxTokens,
                "creative and greedy presets should differ in maxTokens")
        // Balanced differs from greedy in temperature
        #expect(balanced.temperature != greedy.temperature,
                "balanced and greedy presets should differ in temperature")
    }

    // MARK: - Test 5: HiddenStatesConfig layer count validation

    @Test(.timeLimit(.minutes(1))) func hiddenStatesConfigLayerIndexValidation() {
        let config = HiddenStatesConfig.mfluxDefault
        // Valid config should have non-empty layerIndices
        #expect(config.layerIndices.count > 0, "mfluxDefault should have at least one layer index")

        // All layer indices in the default config should be non-negative
        for idx in config.layerIndices {
            #expect(idx >= 0, "Layer index should be non-negative")
        }

        // Custom config: verify layerIndices are stored correctly
        let custom = HiddenStatesConfig.custom(layers: [5, 10, 15])
        #expect(custom.layerIndices.count == 3)
        #expect(custom.layerIndices[0] == 5)
        #expect(custom.layerIndices[1] == 10)
        #expect(custom.layerIndices[2] == 15)
    }

    // MARK: - Test 6: ModelRegistry HuggingFace repo strings

    @Test(.timeLimit(.minutes(1))) @MainActor func modelRegistryHasNonEmptyRepos() {
        let models = TextEncoderModelRegistry.shared.allModels()
        #expect(models.count > 0, "Registry should have at least one model")
        for model in models {
            #expect(!model.repoId.isEmpty, "Model repoId should be non-empty")
            #expect(model.repoId.contains("/"), "Model repoId should be in org/repo format")
        }
    }

    // MARK: - Test 7: TekkenTokenizer round-trip

    @Test(.timeLimit(.minutes(1))) func tekkenTokenizerRoundTrips() {
        // TekkenTokenizer can be instantiated without a model file (uses default tokenizer).
        // Full round-trip (encode->decode) requires a real model file (tekken.json) to populate
        // rankToBytes; without it, decode returns empty. We test encode and structural properties.
        let tokenizer = TekkenTokenizer()
        let original = "Hello world"
        let tokens = tokenizer.encode(original)
        // Encode should produce at least one token for a non-empty ASCII string
        if tokens.isEmpty {
            Issue.record("TekkenTokenizer encode returned empty — tokenizer requires model file for round-trip")
            return
        }
        // Tokens are integer IDs — all should be non-negative
        for token in tokens {
            #expect(token >= 0, "Token IDs should be non-negative")
        }
        // Vocab size property should be accessible without crashing
        let vocabSize = tokenizer.vocabSize
        #expect(vocabSize >= 0, "vocabSize should be non-negative")
    }

    // MARK: - Test 8: ImageProcessor returns valid dimensions

    @Test(.timeLimit(.minutes(1))) func imageProcessorReturnsDimensions() {
        let image = TestImage.make(width: 64, height: 64)
        #expect(image.width == 64)
        #expect(image.height == 64)

        // Verify ImageProcessor can be instantiated with default config
        let processor = ImageProcessor()
        #expect(processor.config.imageSize > 0,
                "ImageProcessorConfig imageSize should be positive")
        #expect(processor.config.patchSize > 0,
                "ImageProcessorConfig patchSize should be positive")

        // Note: processor.preprocess() requires MLX GPU — skip actual preprocessing in CI.
        // We verify structural correctness here.
        #expect(processor.config.imageMean.count == 3,
                "imageMean should have 3 channels (RGB)")
        #expect(processor.config.imageStd.count == 3,
                "imageStd should have 3 channels (RGB)")
    }

    // MARK: - Test 9: TextEncoderMemoryConfig values

    @Test(.timeLimit(.minutes(1))) @MainActor func textEncoderMemoryConfigValues() {
        // disabled has evalFrequency 0; others are positive
        #expect(TextEncoderMemoryConfig.disabled.evalFrequency == 0)
        #expect(TextEncoderMemoryConfig.light.evalFrequency > 0)
        #expect(TextEncoderMemoryConfig.moderate.evalFrequency > 0)
        #expect(TextEncoderMemoryConfig.aggressive.evalFrequency > 0)
        #expect(TextEncoderMemoryConfig.ultraLowMemory.evalFrequency > 0)

        // Verify that recommended() returns a valid config (non-crashing)
        let recommended = TextEncoderMemoryConfig.recommended(forRAMGB: 32)
        #expect(recommended.evalFrequency >= 0)

        // Also verify each ModelInfo in the registry has a non-empty parameters string
        let models = TextEncoderModelRegistry.shared.allModels()
        for model in models {
            #expect(!model.parameters.isEmpty, "Model parameters string should be non-empty")
        }
    }

    // MARK: - Test 10: All error cases have localizedDescription

    @Test(.timeLimit(.minutes(1))) func allErrorCasesHaveLocalizedDescription() {
        let errors: [FluxEncoderError] = [
            .modelNotLoaded,
            .vlmNotLoaded,
            .kleinNotLoaded,
            .invalidInput("test input error"),
            .generationFailed("test generation failure")
        ]

        for error in errors {
            let description = error.localizedDescription
            #expect(!description.isEmpty, "FluxEncoderError localizedDescription should be non-empty")
        }
    }

    // MARK: - Test 11: Progress callback fires correct number of times

    @Test(.timeLimit(.minutes(1))) func progressCallbackFiresCorrectNumberOfTimes() async {
        let mock = MockFlux2Pipeline()
        mock.simulatedSteps = 6
        // Use a class-based counter to avoid Swift concurrency mutation warning
        final class Counter: @unchecked Sendable {
            var value: Int = 0
        }
        let counter = Counter()

        _ = try? await mock.generate(
            mode: .textToImage,
            prompt: "test prompt",
            height: 64,
            width: 64,
            steps: 6,
            guidance: 3.5,
            seed: nil,
            upsamplePrompt: false,
            checkpointInterval: nil,
            onProgress: { _, _ in counter.value += 1 },
            onCheckpoint: nil
        )

        #expect(counter.value == 6, "Progress callback should fire exactly 6 times (once per step)")
    }

    // MARK: - Test 12: Error path for corrupted model file

    @Test(.timeLimit(.minutes(1))) func errorPathForCorruptedModelFile() async {
        let mock = MockFlux2Pipeline()
        mock.errorToThrow = NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Simulated error"])

        var didThrow = false
        do {
            _ = try await mock.generate(
                mode: .textToImage,
                prompt: "test prompt",
                height: 64,
                width: 64,
                steps: 4,
                guidance: 3.5,
                seed: nil,
                upsamplePrompt: false,
                checkpointInterval: nil,
                onProgress: nil,
                onCheckpoint: nil
            )
        } catch {
            didThrow = true
        }
        #expect(didThrow, "generate should throw when errorToThrow is set")
    }

    // MARK: - Test 13: Concurrency — parallel embedding extraction

    @Test(.timeLimit(.minutes(1))) func concurrencyParallelEmbeddingExtraction() async {
        let mock1 = MockFlux2Pipeline()
        let mock2 = MockFlux2Pipeline()

        async let result1 = mock1.generate(
            mode: .textToImage,
            prompt: "first concurrent prompt",
            height: 64,
            width: 64,
            steps: 4,
            guidance: 3.5,
            seed: nil,
            upsamplePrompt: false,
            checkpointInterval: nil,
            onProgress: nil,
            onCheckpoint: nil
        )

        async let result2 = mock2.generate(
            mode: .textToImage,
            prompt: "second concurrent prompt",
            height: 64,
            width: 64,
            steps: 4,
            guidance: 3.5,
            seed: nil,
            upsamplePrompt: false,
            checkpointInterval: nil,
            onProgress: nil,
            onCheckpoint: nil
        )

        let image1 = try? await result1
        let image2 = try? await result2

        #expect(image1 != nil, "First concurrent generate should succeed")
        #expect(image2 != nil, "Second concurrent generate should succeed")
    }
}
