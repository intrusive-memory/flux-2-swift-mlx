/**
 * ConfigurationTests.swift
 * Unit tests for configuration structures
 */

import Testing
@testable import FluxTextEncoders

@Suite("ConfigurationTests")
struct ConfigurationTests {

    // MARK: - MistralTextConfig Tests

    @Test func mistralTextConfigDefaults() {
        let config = MistralTextConfig.mistralSmall32

        // Verify Mistral Small 3.2 defaults
        #expect(config.vocabSize == 131_072, "Vocab size should be 131K")
        #expect(config.hiddenSize == 5120, "Hidden size should be 5120")
        #expect(config.intermediateSize == 14336, "Intermediate size should be 14336")
        #expect(config.numHiddenLayers == 40, "Should have 40 layers")
        #expect(config.numAttentionHeads == 32, "Should have 32 attention heads")
        #expect(config.numKeyValueHeads == 8, "Should have 8 KV heads (GQA)")
        #expect(config.maxPositionEmbeddings == 131_072, "Max position should be 131K")
        #expect(config.headDim == 128, "Head dimension should be 128")
    }

    @Test func mistralTextConfigCustomInit() {
        let config = MistralTextConfig(
            vocabSize: 50000,
            hiddenSize: 1024,
            intermediateSize: 4096,
            numHiddenLayers: 12,
            numAttentionHeads: 16,
            numKeyValueHeads: 4,
            maxPositionEmbeddings: 8192
        )

        #expect(config.vocabSize == 50000)
        #expect(config.hiddenSize == 1024)
        #expect(config.numHiddenLayers == 12)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 4)
    }

    @Test func mistralTextConfigRopeSettings() {
        let config = MistralTextConfig.mistralSmall32

        #expect(config.ropeTheta == 1_000_000.0, "RoPE theta should be 1M")
        #expect(config.rmsNormEps == 1e-5, "RMS norm eps should be 1e-5")
    }

    @Test func mistralTextConfigActivation() {
        let config = MistralTextConfig.mistralSmall32

        #expect(config.hiddenAct == "silu", "Activation should be silu")
        #expect(!config.attentionBias, "Attention bias should be false")
        #expect(!config.mlpBias, "MLP bias should be false")
    }

    // MARK: - MistralVisionConfig Tests

    @Test func mistralVisionConfigDefaults() {
        let config = MistralVisionConfig.defaultVision

        #expect(config.hiddenSize == 1024, "Vision hidden size should be 1024")
        #expect(config.imageSize == 384, "Image size should be 384")
        #expect(config.patchSize == 14, "Patch size should be 14")
        #expect(config.numChannels == 3, "Should have 3 color channels")
        #expect(config.numHiddenLayers == 24, "Vision should have 24 layers")
        #expect(config.numAttentionHeads == 16, "Vision should have 16 attention heads")
    }

    @Test func mistralVisionConfigCustomInit() {
        let config = MistralVisionConfig(
            hiddenSize: 768,
            imageSize: 224,
            patchSize: 16,
            numChannels: 3,
            numHiddenLayers: 12,
            numAttentionHeads: 12,
            intermediateSize: 3072
        )

        #expect(config.hiddenSize == 768)
        #expect(config.imageSize == 224)
        #expect(config.patchSize == 16)
    }

    // MARK: - MistralConfig Tests

    @Test func mistralConfigInit() {
        let textConfig = MistralTextConfig.mistralSmall32
        let config = MistralConfig(textConfig: textConfig, modelType: "mistral")

        #expect(config.textConfig.vocabSize == 131_072)
        #expect(config.modelType == "mistral")
        #expect(config.visionConfig == nil, "Vision config should be nil for text-only")
    }

    @Test func mistralConfigWithVision() {
        let textConfig = MistralTextConfig.mistralSmall32
        let visionConfig = MistralVisionConfig.defaultVision
        let config = MistralConfig(
            textConfig: textConfig,
            visionConfig: visionConfig,
            modelType: "mistral_vlm"
        )

        #expect(config.visionConfig != nil, "Vision config should not be nil")
        #expect(config.visionConfig?.imageSize == 384)
    }

    // MARK: - GenerationConfig Tests

    @Test func generationConfigDefaults() {
        let config = GenerationConfig.mistralDefault

        #expect(config.bosTokenId == 1, "BOS token should be 1")
        #expect(config.eosTokenId == 2, "EOS token should be 2")
        #expect(config.padTokenId == nil, "Default pad token should be nil")
    }

    @Test func generationConfigCustomInit() {
        let config = GenerationConfig(bosTokenId: 0, eosTokenId: 1, padTokenId: 2)

        #expect(config.bosTokenId == 0)
        #expect(config.eosTokenId == 1)
        #expect(config.padTokenId == 2)
    }

    // MARK: - GenerateParameters Tests

    @Test func generateParametersDefaults() {
        let params = GenerateParameters()

        #expect(params.maxTokens == 2048, "Default max tokens should be 2048")
        #expect(params.temperature == 0.7, "Default temperature should be 0.7")
        #expect(params.topP == 0.95, "Default topP should be 0.95")
        #expect(params.repetitionPenalty == 1.1, "Default repetition penalty should be 1.1")
        #expect(params.seed == nil, "Default seed should be nil")
    }

    @Test func generateParametersGreedyPreset() {
        let params = GenerateParameters.greedy

        #expect(params.temperature == 0.0, "Greedy should have temperature 0")
        #expect(params.topP == 1.0, "Greedy should have topP 1.0")
        #expect(params.repetitionPenalty == 1.0, "Greedy should have no repetition penalty")
    }

    @Test func generateParametersCreativePreset() {
        let params = GenerateParameters.creative

        #expect(params.temperature == 0.9, "Creative should have high temperature")
        #expect(params.maxTokens == 4096, "Creative should have more tokens")
        #expect(params.repetitionPenalty == 1.2, "Creative should have higher repetition penalty")
    }

    @Test func generateParametersBalancedPreset() {
        let params = GenerateParameters.balanced

        #expect(params.temperature == 0.7, "Balanced temperature should be 0.7")
        #expect(params.topP == 0.9, "Balanced topP should be 0.9")
        #expect(params.maxTokens == 2048, "Balanced max tokens should be 2048")
        #expect(params.repetitionPenalty == 1.1, "Balanced repetition penalty should be 1.1")
    }

    @Test func generateParametersCustomInit() {
        let params = GenerateParameters(
            maxTokens: 100,
            temperature: 0.5,
            topP: 0.8,
            repetitionPenalty: 1.5,
            repetitionContextSize: 50,
            seed: 42
        )

        #expect(params.maxTokens == 100)
        #expect(params.temperature == 0.5)
        #expect(params.topP == 0.8)
        #expect(params.repetitionPenalty == 1.5)
        #expect(params.repetitionContextSize == 50)
        #expect(params.seed == 42)
    }

    @Test func generateParametersMaxContextLength() {
        #expect(GenerateParameters.maxContextLength == 131_072,
                      "Max context should be 131K for Mistral Small 3.2")
    }

}
