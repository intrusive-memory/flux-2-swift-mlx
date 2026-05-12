// Flux2TelemetryWeightLoadHistogramTests.swift
// Tests for Flux2WeightLoader.dtypeHistogram(_:) — no real model downloads.
// All tensors are synthesised in-memory from known dtypes so these tests are
// safe to run in CI without a local model cache.

import MLX
import Testing

@testable import Flux2Core

@Suite("Flux2TelemetryWeightLoadHistogramTests")
struct Flux2TelemetryWeightLoadHistogramTests {

    // MARK: - Helpers

    /// Build a small tensor with a given element type and flat size.
    private static func zeros<T: HasDType>(_ shape: [Int], type: T.Type) -> MLXArray {
        MLXArray.zeros(shape, type: T.self)
    }

    // MARK: - Tests

    /// Empty param dict → empty histogram.
    @Test func emptyParamsProducesEmptyHistogram() {
        let result = Flux2WeightLoader.dtypeHistogram([:])
        #expect(result.isEmpty, "Expected empty histogram for empty params, got \(result)")
    }

    /// Single float16 tensor of shape [4, 4] → {"float16": 16}.
    @Test func singleFloat16TensorHistogram() {
        let params: [String: MLXArray] = [
            "layer.weight": Self.zeros([4, 4], type: Float16.self),
        ]
        let result = Flux2WeightLoader.dtypeHistogram(params)
        #expect(result["float16"] == 16, "Expected float16 count 16, got \(result)")
        #expect(result.count == 1, "Expected exactly one dtype key, got \(result)")
    }

    /// Mixed float16 (4×4 = 16 elements) + float32 (2×2 = 4 elements)
    /// → {"float16": 16, "float32": 4}.
    @Test func mixedDtypesHistogram() {
        let params: [String: MLXArray] = [
            "transformer.weight": Self.zeros([4, 4], type: Float16.self),  // 16 scalars
            "vae.bias":           Self.zeros([2, 2], type: Float32.self),   //  4 scalars
        ]
        let result = Flux2WeightLoader.dtypeHistogram(params)
        #expect(result["float16"] == 16, "Expected float16 count 16, got \(result)")
        #expect(result["float32"] == 4,  "Expected float32 count 4, got \(result)")
        #expect(result.count == 2, "Expected exactly two dtype keys, got \(result)")
    }

    /// Multiple tensors of the same dtype accumulate into one bucket.
    @Test func accumulationWithinSameDtype() {
        let params: [String: MLXArray] = [
            "a": Self.zeros([3],    type: Float16.self),  //  3
            "b": Self.zeros([5, 2], type: Float16.self),  // 10
            "c": Self.zeros([1],    type: Float16.self),  //  1
        ]
        let result = Flux2WeightLoader.dtypeHistogram(params)
        #expect(result["float16"] == 14, "Expected float16 count 14, got \(result)")
        #expect(result.count == 1)
    }

    /// scalar (shape []) tensors count as 1 element each.
    @Test func scalarTensorCountsAsOneElement() {
        let params: [String: MLXArray] = [
            "scale": MLXArray(Float16(1.0)),  // shape [] → 1 element
        ]
        let result = Flux2WeightLoader.dtypeHistogram(params)
        #expect(result["float16"] == 1, "Expected scalar to contribute 1 element, got \(result)")
    }
}
