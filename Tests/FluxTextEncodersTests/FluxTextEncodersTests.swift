/**
 * FluxTextEncodersTests.swift
 * Main unit tests for FluxTextEncoders library
 */

import TestHelpers
import Testing

@testable import FluxTextEncoders

@Suite("FluxTextEncodersTests")
struct FluxTextEncodersTests {

  // MARK: - Error Tests

  @Test func fluxEncoderErrorDescriptions() {
    #expect(
      FluxEncoderError.modelNotLoaded.errorDescription
        == "Model not loaded. Call loadModel() first.")

    #expect(
      FluxEncoderError.vlmNotLoaded.errorDescription
        == "VLM not loaded. Call loadVLMModel() first for vision capabilities.")

    let invalidInput = FluxEncoderError.invalidInput("test message")
    #expect(invalidInput.errorDescription == "Invalid input: test message")

    let genFailed = FluxEncoderError.generationFailed("gen error")
    #expect(genFailed.errorDescription == "Generation failed: gen error")
  }

  // MARK: - Not-Loaded Guards
  //
  // Nothing in this test process loads a model into the shared singleton, so
  // every public entry point must fail fast with the matching "not loaded"
  // error instead of touching a nil model.

  private func expectNotLoaded(
    _ expected: FluxEncoderError,
    sourceLocation: SourceLocation = #_sourceLocation,
    _ body: () throws -> Void
  ) {
    do {
      try body()
      Issue.record("Expected \(expected) but the call succeeded", sourceLocation: sourceLocation)
    } catch let error as FluxEncoderError {
      #expect(
        error.errorDescription == expected.errorDescription, sourceLocation: sourceLocation)
    } catch {
      Issue.record("Unexpected error \(error)", sourceLocation: sourceLocation)
    }
  }

  @Test func generateThrowsWhenModelNotLoaded() {
    expectNotLoaded(.modelNotLoaded) {
      _ = try FluxTextEncoders.shared.generate(prompt: "a cat")
    }
  }

  @Test func chatThrowsWhenModelNotLoaded() {
    expectNotLoaded(.modelNotLoaded) {
      _ = try FluxTextEncoders.shared.chat(messages: [["role": "user", "content": "hi"]])
    }
  }

  @Test func extractEmbeddingsThrowsWhenModelNotLoaded() {
    expectNotLoaded(.modelNotLoaded) {
      _ = try FluxTextEncoders.shared.extractEmbeddings(prompt: "a cat")
    }
  }

  @Test func analyzeImageThrowsWhenVLMNotLoaded() {
    let image = TestImage.make(width: 8, height: 8)
    expectNotLoaded(.vlmNotLoaded) {
      _ = try FluxTextEncoders.shared.analyzeImage(image: image, prompt: "describe")
    }
  }

  @Test func singletonStartsUnloaded() {
    #expect(!FluxTextEncoders.shared.isModelLoaded)
    #expect(!FluxTextEncoders.shared.isVLMLoaded)
    #expect(!FluxTextEncoders.shared.isKleinLoaded)
  }

  // MARK: - Tokenization Tests (Basic)

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
      #expect(
        layerIdx < maxLayer, "Layer index \(layerIdx) should be less than num layers \(maxLayer)")
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
  @Test func tokenizerVariousInputs() throws {
    let tokenizer = TekkenTokenizer()

    // ASCII
    let asciiTokens = try tokenizer.encode("Hello")
    #expect(!asciiTokens.isEmpty)

    // Unicode
    let unicodeTokens = try tokenizer.encode("世界")
    #expect(!unicodeTokens.isEmpty)

    // Mixed
    let mixedTokens = try tokenizer.encode("Hello 世界!")
    #expect(!mixedTokens.isEmpty)

    // Numbers
    let numberTokens = try tokenizer.encode("12345")
    #expect(!numberTokens.isEmpty)

    // Special chars
    let specialTokens = try tokenizer.encode("@#$%^&*()")
    #expect(!specialTokens.isEmpty)
  }

  /// Test chat template produces valid structure
  @Test func chatTemplateStructure() {
    let tokenizer = TekkenTokenizer()

    let messages: [[String: String]] = [
      ["role": "system", "content": "You are helpful"],
      ["role": "user", "content": "Hello"],
      ["role": "assistant", "content": "Hi there!"],
      ["role": "user", "content": "How are you?"],
    ]

    let prompt = tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false)

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
