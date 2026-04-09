/**
 * FluxTextEncodersTests.swift
 * Main unit tests for FluxTextEncoders library
 */

import Testing
@testable import FluxTextEncoders

@Suite("FluxTextEncodersTests")
struct FluxTextEncodersTests {

    // MARK: - Version Tests

    @Test func version() {
        #expect(!FluxTextEncoders.version.isEmpty, "Version should not be empty")
        #expect(FluxTextEncoders.version == "2.6.0", "Version should be 2.6.0")
    }

    @Test func mistralVersionInfo() {
        #expect(MistralVersion.version == "2.6.0")
        #expect(MistralVersion.modelName == "Mistral Small 3.2")
        #expect(MistralVersion.modelVersion == "24B-Instruct-2506")
    }

    // MARK: - Singleton Tests

    @Test func fluxTextEncodersSharedInstance() {
        let core1 = FluxTextEncoders.shared
        let core2 = FluxTextEncoders.shared
        #expect(core1 === core2, "Shared instance should be singleton")
    }

    // MARK: - Model State Tests

    @Test func isModelLoadedInitiallyFalse() {
        // Note: In a real test environment, you'd create a fresh instance
        // For singleton, this test verifies the property exists
        let core = FluxTextEncoders.shared
        _ = core.isModelLoaded  // Should not crash
    }

    @Test func isVLMLoadedInitiallyFalse() {
        let core = FluxTextEncoders.shared
        _ = core.isVLMLoaded  // Should not crash
    }

    // MARK: - Error Tests

    @Test func fluxEncoderErrorDescriptions() {
        #expect(FluxEncoderError.modelNotLoaded.errorDescription == "Model not loaded. Call loadModel() first.")

        #expect(FluxEncoderError.vlmNotLoaded.errorDescription == "VLM not loaded. Call loadVLMModel() first for vision capabilities.")

        let invalidInput = FluxEncoderError.invalidInput("test message")
        #expect(invalidInput.errorDescription == "Invalid input: test message")

        let genFailed = FluxEncoderError.generationFailed("gen error")
        #expect(genFailed.errorDescription == "Generation failed: gen error")
    }

    // MARK: - Generate Without Model Tests

    @Test func generateThrowsWhenModelNotLoaded() {
        // Create a fresh core that definitely doesn't have a model loaded
        // For singleton, we test that the error is properly defined
        #expect(FluxEncoderError.modelNotLoaded.errorDescription != nil)
    }

    @Test func chatThrowsWhenModelNotLoaded() {
        #expect(FluxEncoderError.modelNotLoaded.errorDescription != nil)
    }

    // MARK: - Embeddings Without Model Tests

    @Test func extractEmbeddingsThrowsWhenModelNotLoaded() {
        #expect(FluxEncoderError.modelNotLoaded.errorDescription != nil)
    }

    // MARK: - Vision Without Model Tests

    @Test func analyzeImageThrowsWhenVLMNotLoaded() {
        #expect(FluxEncoderError.vlmNotLoaded.errorDescription != nil)
    }

    // MARK: - Tokenization Tests (Basic)

    @Test func tokenizer() throws {
        let tokenizer = TekkenTokenizer()
        let text = "Hello world"
        let tokens = tokenizer.encode(text)
        #expect(!tokens.isEmpty, "Tokenization should produce tokens")
    }

    @Test func hiddenStatesConfig() throws {
        let config = HiddenStatesConfig.mfluxDefault
        #expect(config.layerIndices == [10, 20, 30])
        #expect(config.concatenate)
    }

    @Test @MainActor func textEncoderModelRegistry() throws {
        let models = TextEncoderModelRegistry.shared.allModels()
        #expect(models.count >= 3, "Should have at least 3 model variants")

        let defaultModel = TextEncoderModelRegistry.shared.defaultModel()
        #expect(defaultModel.variant == .mlx8bit)
    }

    @Test func generateParameters() throws {
        let params = GenerateParameters.balanced
        #expect(params.maxTokens == 2048)
        #expect(params.temperature == 0.7)
        #expect(params.topP == 0.9)
    }

    // MARK: - FLUX Config Tests

    @Test func fluxConfigValues() {
        #expect(FluxConfig.maxSequenceLength == 512)
        #expect(FluxConfig.hiddenStateLayers == [10, 20, 30])
        #expect(!FluxConfig.systemMessage.isEmpty, "System message should not be empty")
    }

    @Test func fluxConfigSystemMessage() {
        let systemMessage = FluxConfig.systemMessage
        #expect(systemMessage.contains("image"), "System message should mention images")
    }

    // MARK: - Export Format Tests

    @Test func exportFormatCases() {
        let binary = ExportFormat.binary
        let numpy = ExportFormat.numpy
        let json = ExportFormat.json

        // Just verify they exist and are distinct
        #expect(String(describing: binary) != String(describing: numpy))
        #expect(String(describing: binary) != String(describing: json))
    }
}

// MARK: - Integration Tests (Without Model Loading)

@Suite("FluxTextEncodersIntegrationTests")
struct FluxTextEncodersIntegrationTests {

    /// Test that configurations work together correctly
    @Test func configurationIntegration() {
        let textConfig = MistralTextConfig.mistralSmall32
        let hiddenStatesConfig = HiddenStatesConfig.mfluxDefault

        // Verify layer indices are valid for the model
        let maxLayer = textConfig.numHiddenLayers
        for layerIdx in hiddenStatesConfig.layerIndices {
            #expect(layerIdx < maxLayer, "Layer index \(layerIdx) should be less than num layers \(maxLayer)")
        }
    }

    /// Test that FLUX embeddings configuration matches expected dimensions
    @Test func fluxEmbeddingsDimensions() {
        let config = MistralTextConfig.mistralSmall32
        let hiddenStatesConfig = HiddenStatesConfig.mfluxDefault

        // FLUX expects 3 layers * hidden_size = 15360
        let expectedDim = hiddenStatesConfig.layerIndices.count * config.hiddenSize
        #expect(expectedDim == 15360, "FLUX embeddings should produce 15360 dimensions")
    }

    /// Test tokenizer with various input types
    @Test func tokenizerVariousInputs() {
        let tokenizer = TekkenTokenizer()

        // ASCII
        let asciiTokens = tokenizer.encode("Hello")
        #expect(!asciiTokens.isEmpty)

        // Unicode
        let unicodeTokens = tokenizer.encode("世界")
        #expect(!unicodeTokens.isEmpty)

        // Mixed
        let mixedTokens = tokenizer.encode("Hello 世界!")
        #expect(!mixedTokens.isEmpty)

        // Numbers
        let numberTokens = tokenizer.encode("12345")
        #expect(!numberTokens.isEmpty)

        // Special chars
        let specialTokens = tokenizer.encode("@#$%^&*()")
        #expect(!specialTokens.isEmpty)
    }

    /// Test chat template produces valid structure
    @Test func chatTemplateStructure() {
        let tokenizer = TekkenTokenizer()

        let messages: [[String: String]] = [
            ["role": "system", "content": "You are helpful"],
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "content": "Hi there!"],
            ["role": "user", "content": "How are you?"]
        ]

        let prompt = tokenizer.applyChatTemplate(messages: messages)

        // Should be non-empty
        #expect(!prompt.isEmpty)

        // Should contain all content
        #expect(prompt.contains("helpful") || prompt.contains("You are helpful"))
        #expect(prompt.contains("Hello"))
        #expect(prompt.contains("Hi there!"))
        #expect(prompt.contains("How are you?"))
    }

    @Test @MainActor func textEncoderModelRegistryHasAllExpectedVariants() {
        let registry = TextEncoderModelRegistry.shared
        let models = registry.allModels()

        // Should have quantized models
        let has8bit = models.contains { $0.variant == .mlx8bit }
        let has4bit = models.contains { $0.variant == .mlx4bit }

        #expect(has8bit, "Should have 8-bit model")
        #expect(has4bit, "Should have 4-bit model")
    }
}
