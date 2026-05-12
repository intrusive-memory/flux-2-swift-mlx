// Flux2TelemetryWeightLoadHistogramTests.swift — Sortie 7a contract tests for
// `Flux2WeightLoader.dtypeHistogram(_:)`.
//
// F8: imports `TestHelpers` so `MockTelemetryReporter` is in scope for sibling
//     test files (not directly used here, but the import matches the campaign
//     convention so callers don't have to re-import per file).
// F9: uses the STATIC method `MLXArray.zeros(_:type:)`, NEVER the initializer
//     form `MLXArray(zeros: ...)` (which does not exist on `MLXArray`).
// F10: every helper in this file is a non-static instance method on the suite
//     struct, so Swift 6 strict-mode static-member access via generics is moot.

import Foundation
import MLX
import Testing
import TestHelpers

@testable import Flux2Core

@Suite("Flux2WeightLoader.dtypeHistogram")
struct Flux2TelemetryWeightLoadHistogramTests {

    // F10: NON-static helpers. Do NOT mark these `private static func` — Swift 6
    // strict mode would then require every call site to be `Self.makeFloat32Tensor()`
    // and a missed call site is a build error.

    private func makeFloat32Tensor() -> MLXArray {
        // F9: `MLXArray.zeros(_:type:)` is the static factory method on the
        // `MLXArray` type. There is no `MLXArray(zeros:)` initializer.
        MLXArray.zeros([2, 2], type: Float32.self)
    }

    private func makeFloat16Tensor() -> MLXArray {
        MLXArray.zeros([4], type: Float16.self)
    }

    private func makeInt32Tensor() -> MLXArray {
        MLXArray.zeros([3, 3], type: Int32.self)
    }

    @Test("empty input returns empty histogram")
    func testEmptyInput() {
        let histogram = Flux2WeightLoader.dtypeHistogram([:])
        #expect(histogram.isEmpty)
    }

    @Test("single dtype bucketed correctly")
    func testSingleDtype() {
        let params: [String: MLXArray] = [
            "layer1.weight": makeFloat32Tensor(),
            "layer1.bias": makeFloat32Tensor(),
        ]
        let histogram = Flux2WeightLoader.dtypeHistogram(params)
        #expect(histogram["float32"] == 2)
        #expect(histogram.count == 1)
    }

    @Test("mixed dtypes are bucketed independently")
    func testMixedDtypes() {
        let params: [String: MLXArray] = [
            "a.weight": makeFloat32Tensor(),
            "b.weight": makeFloat16Tensor(),
            "c.weight": makeFloat16Tensor(),
            "d.indices": makeInt32Tensor(),
        ]
        let histogram = Flux2WeightLoader.dtypeHistogram(params)
        #expect(histogram["float32"] == 1)
        #expect(histogram["float16"] == 2)
        #expect(histogram["int32"] == 1)
        #expect(histogram.count == 3)
    }

    @Test("scalar tensors are counted by dtype")
    func testScalarTensor() {
        // F9: a 0-d (scalar) tensor uses the same static factory; just an empty
        // shape array.
        let scalar = MLXArray.zeros([], type: Float32.self)
        let histogram = Flux2WeightLoader.dtypeHistogram(["scalar": scalar])
        #expect(histogram["float32"] == 1)
        #expect(histogram.count == 1)
    }
}
