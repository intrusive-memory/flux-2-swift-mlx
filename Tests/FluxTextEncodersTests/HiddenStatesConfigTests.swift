/**
 * HiddenStatesConfigTests.swift
 * Unit tests for HiddenStatesConfig
 */

import Testing
@testable import FluxTextEncoders

@Suite("HiddenStatesConfigTests")
struct HiddenStatesConfigTests {

    // MARK: - Preset Tests

    @Test func mfluxDefaultPreset() {
        let config = HiddenStatesConfig.mfluxDefault

        #expect(config.layerIndices == [10, 20, 30],
                      "mflux default should extract from layers 10, 20, 30")
        #expect(config.concatenate,
                     "mflux default should concatenate layers")
        #expect(!config.normalize,
                      "mflux default should not normalize")
        #expect(config.pooling == .none,
                      "mflux default should have no pooling")
    }

    @Test func lastLayerOnlyPreset() {
        let config = HiddenStatesConfig.lastLayerOnly

        #expect(config.layerIndices == [-1],
                      "lastLayerOnly should use index -1")
        #expect(!config.concatenate,
                      "lastLayerOnly should not concatenate (only one layer)")
        #expect(!config.normalize,
                      "lastLayerOnly should not normalize by default")
        #expect(config.pooling == .lastToken,
                      "lastLayerOnly should use lastToken pooling")
    }

    @Test func middleLayerPreset() {
        let config = HiddenStatesConfig.middleLayer

        #expect(config.layerIndices == [20],
                      "middleLayer should extract from layer 20")
        #expect(!config.concatenate)
        #expect(config.pooling == .lastToken)
    }

    @Test func allLayersPreset() {
        let config = HiddenStatesConfig.allLayers

        #expect(config.layerIndices.count == 40,
                      "allLayers should have 40 layer indices")
        #expect(config.layerIndices.first == 0,
                      "allLayers should start at 0")
        #expect(config.layerIndices.last == 39,
                      "allLayers should end at 39")
        #expect(!config.concatenate,
                      "allLayers should not concatenate (too large)")
        #expect(config.pooling == .none)
    }

    // MARK: - Custom Config Tests

    @Test func customInit() {
        let config = HiddenStatesConfig(
            layerIndices: [5, 15, 25],
            concatenate: false,
            normalize: true,
            pooling: .mean
        )

        #expect(config.layerIndices == [5, 15, 25])
        #expect(!config.concatenate)
        #expect(config.normalize)
        #expect(config.pooling == .mean)
    }

    @Test func customBuilder() {
        let config = HiddenStatesConfig.custom(
            layers: [0, 10, 20, 30, 39],
            concatenate: true,
            normalize: true,
            pooling: .lastToken
        )

        #expect(config.layerIndices.count == 5)
        #expect(config.concatenate)
        #expect(config.normalize)
        #expect(config.pooling == .lastToken)
    }

    @Test func customBuilderDefaults() {
        let config = HiddenStatesConfig.custom(layers: [10])

        // Check defaults
        #expect(config.concatenate, "Default concatenate should be true")
        #expect(!config.normalize, "Default normalize should be false")
        #expect(config.pooling == .none, "Default pooling should be none")
    }

    // MARK: - Pooling Strategy Tests

    @Test func poolingStrategyNone() {
        #expect(PoolingStrategy.none.rawValue == "none")
    }

    @Test func poolingStrategyLastToken() {
        #expect(PoolingStrategy.lastToken.rawValue == "lastToken")
    }

    @Test func poolingStrategyMean() {
        #expect(PoolingStrategy.mean.rawValue == "mean")
    }

    @Test func poolingStrategyMax() {
        #expect(PoolingStrategy.max.rawValue == "max")
    }

    @Test func poolingStrategyCLS() {
        #expect(PoolingStrategy.cls.rawValue == "cls")
    }

    // MARK: - Edge Cases

    @Test func emptyLayerIndices() {
        let config = HiddenStatesConfig(
            layerIndices: [],
            concatenate: true,
            normalize: false,
            pooling: .none
        )

        #expect(config.layerIndices.isEmpty)
    }

    @Test func negativeLayerIndices() {
        let config = HiddenStatesConfig(
            layerIndices: [-1, -2, -3],
            concatenate: true,
            normalize: false,
            pooling: .none
        )

        // Negative indices should be preserved (resolved at extraction time)
        #expect(config.layerIndices == [-1, -2, -3])
    }

    @Test func singleLayerWithConcatenate() {
        let config = HiddenStatesConfig(
            layerIndices: [10],
            concatenate: true,  // Should work even with single layer
            normalize: false,
            pooling: .none
        )

        #expect(config.layerIndices.count == 1)
        #expect(config.concatenate)
    }

    // MARK: - FLUX.2 Compatibility Tests

    @Test func fluxCompatibleDimensions() {
        let config = HiddenStatesConfig.mfluxDefault
        let hiddenSize = 5120  // Mistral Small 3.2 hidden size
        let numLayers = config.layerIndices.count

        // FLUX.2 expects 15360 dimensions (3 layers * 5120)
        let expectedDimension = numLayers * hiddenSize
        #expect(expectedDimension == 15360,
                      "FLUX.2 config should produce 15360 dimensions")
    }
}
