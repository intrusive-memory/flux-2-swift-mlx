// Flux2TelemetryAnomalyTests.swift — Sortie 7b direct unit tests for
// `Flux2AnomalyDetector.anomalies(in:checkZeroLatent:expectedDtype:)` plus
// a synthetic event-pair test verifying the NaN anomaly emission contract.
//
// Option A "pure contract" test: exercises `Flux2AnomalyDetector` directly
// and uses `MockTelemetryReporter` for the synthetic pair test. No real model,
// no real MLX kernels.
//
// F4: outOfRange test drives stat values using TuberiaTensorStat.defaultOutOfRangeThreshold
//     — never a literal.
// F8: `import TestHelpers` brings `MockTelemetryReporter` into scope.
// F10: every helper is a non-static instance method.

import Foundation
import Testing
import TestHelpers
import Tuberia

@testable import Flux2Core

@Suite("Flux2AnomalyDetector unit + numericalAnomaly pair contract")
struct Flux2TelemetryAnomalyTests {

    // F10: NON-static helpers. Matches the TuberiaTensorStat init order:
    // (shape, dtype, min, max, mean, std, hasNaN, hasInf).

    private func clean() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 16],
            dtype: "float16",
            min: -1.0,
            max: 1.0,
            mean: 0.0,
            std: 1.0,
            hasNaN: false,
            hasInf: false
        )
    }

    private func withNaN() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 16],
            dtype: "float16",
            min: -1.0,
            max: 1.0,
            mean: 0.0,
            std: 1.0,
            hasNaN: true,
            hasInf: false
        )
    }

    private func withInf() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 16],
            dtype: "float16",
            min: -.infinity,
            max: .infinity,
            mean: 0.0,
            std: 1.0,
            hasNaN: false,
            hasInf: true
        )
    }

    // F4: Reference TuberiaTensorStat.defaultOutOfRangeThreshold by NAME,
    // never by literal value.
    private func outOfRange() -> TuberiaTensorStat {
        let high = TuberiaTensorStat.defaultOutOfRangeThreshold * 2.0
        return TuberiaTensorStat(
            shape: [1, 16],
            dtype: "float16",
            min: -high,
            max: high,
            mean: 0.0,
            std: 1.0,
            hasNaN: false,
            hasInf: false
        )
    }

    private func zeroLatent() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 16],
            dtype: "float16",
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            hasNaN: false,
            hasInf: false
        )
    }

    @Test("NaN detected")
    func testNaN() {
        let kinds = Flux2AnomalyDetector.anomalies(in: withNaN())
        #expect(kinds.contains(.nan))
    }

    @Test("Inf detected")
    func testInf() {
        let kinds = Flux2AnomalyDetector.anomalies(in: withInf())
        #expect(kinds.contains(.inf))
    }

    @Test("outOfRange via abs(max/min) > defaultOutOfRangeThreshold (F4)")
    func testOutOfRange() {
        let kinds = Flux2AnomalyDetector.anomalies(in: outOfRange())
        #expect(kinds.contains(.outOfRange))
    }

    @Test("zeroLatent only when checkZeroLatent: true")
    func testZeroLatent() {
        let stat = zeroLatent()
        let withCheck = Flux2AnomalyDetector.anomalies(in: stat, checkZeroLatent: true)
        let withoutCheck = Flux2AnomalyDetector.anomalies(in: stat, checkZeroLatent: false)
        #expect(withCheck.contains(.zeroLatent))
        #expect(!withoutCheck.contains(.zeroLatent))
    }

    @Test("dtypeUnexpected when expectedDtype differs")
    func testDtypeUnexpected() {
        let stat = clean()    // dtype "float16"
        let kinds = Flux2AnomalyDetector.anomalies(in: stat, expectedDtype: "float32")
        #expect(kinds.contains(.dtypeUnexpected))
    }

    @Test("clean stat → no anomalies (negative)")
    func testClean() {
        let kinds = Flux2AnomalyDetector.anomalies(in: clean())
        #expect(kinds.isEmpty)
    }

    @Test("Synthetic pair: NaN at step 2 fires both denoiseStepComplete + numericalAnomaly(.nan)")
    func testEventPair() async {
        let mock = MockTelemetryReporter()
        let cleanS = clean()
        let nanS = withNaN()

        await mock.capture(.denoiseStepComplete(
            variant: .textToImage,
            stepIndex: 2,
            totalSteps: 4,
            sigma: 0.5,
            timestep: 500.0,
            latentBeforeStat: cleanS,
            noisePredStat: nanS,
            latentAfterStat: cleanS,
            kvCacheLayerCount: nil,
            kvCacheHit: nil,
            durationSeconds: 0.1
        ))

        // Emit the same anomaly events the production code would emit: one
        // numericalAnomaly per AnomalyKind in the detected set.
        for kind in Flux2AnomalyDetector.anomalies(in: nanS, checkZeroLatent: true) {
            await mock.capture(.numericalAnomaly(
                phase: "denoiseStepComplete",
                kind: kind,
                stepIndex: 2,
                stat: nanS
            ))
        }

        let events = await mock.events()
        #expect(events.count == 2)

        // First event is denoiseStepComplete:
        if case .denoiseStepComplete = events[0] {
            // ok
        } else {
            Issue.record("expected denoiseStepComplete at index 0, got \(events[0])")
        }

        // Second event is numericalAnomaly(.nan, stepIndex: 2):
        if case let .numericalAnomaly(_, kind, stepIndex, _) = events[1] {
            #expect(kind == .nan)
            #expect(stepIndex == 2)
        } else {
            Issue.record("expected numericalAnomaly(.nan, stepIndex: 2) at index 1, got \(events[1])")
        }
    }
}
