// Flux2TelemetryAnomalyTests.swift
// Contract tests for the numericalAnomaly side-channel and Flux2AnomalyDetector.
//
// Two test groups:
//   1. Synthetic-event pair: a denoiseStepComplete at step 2 carrying a NaN-flagged
//      noisePredStat fires alongside a numericalAnomaly(.nan) event.
//   2. Direct Flux2AnomalyDetector.anomalies(in:) unit tests for all five AnomalyKinds:
//      .nan, .inf, .outOfRange, .zeroLatent, .dtypeUnexpected.
//
// No real weights, no GPU, no Metal required.

import Foundation
import TestHelpers
import Testing
import Tuberia

@testable import Flux2Core

@Suite("Flux2TelemetryAnomalyTests")
struct Flux2TelemetryAnomalyTests {

    // MARK: - Helpers

    private static let defaultShape = [1, 16, 8, 8]
    private static let defaultDtype = "float16"

    /// Build a normal (anomaly-free) stat.
    private static func normalStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: defaultShape,
            dtype: defaultDtype,
            min: -0.3,
            max: 0.3,
            mean: 0.01,
            std: 0.2,
            hasNaN: false,
            hasInf: false
        )
    }

    /// Build a stat that has `hasNaN: true`.
    private static func nanStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: defaultShape,
            dtype: defaultDtype,
            min: -0.3,
            max: 0.3,
            mean: 0.0,
            std: 0.2,
            hasNaN: true,
            hasInf: false
        )
    }

    /// Build a stat that has `hasInf: true`.
    private static func infStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: defaultShape,
            dtype: defaultDtype,
            min: -1e9,
            max: 1e9,
            mean: 0.0,
            std: 5e8,
            hasNaN: false,
            hasInf: true
        )
    }

    /// Build a stat whose max exceeds the default out-of-range threshold (1e6).
    private static func outOfRangeStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: defaultShape,
            dtype: defaultDtype,
            min: -2e6,
            max: 2e6,
            mean: 0.0,
            std: 1e6,
            hasNaN: false,
            hasInf: false
        )
    }

    /// Build a stat representing a collapsed (zero) latent: mean ≈ 0, std ≈ 0.
    private static func zeroLatentStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: defaultShape,
            dtype: defaultDtype,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            hasNaN: false,
            hasInf: false
        )
    }

    /// Build a stat with a dtype mismatch relative to an expected value.
    private static func dtypeMismatchStat(actual: String = "float32") -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: defaultShape,
            dtype: actual,      // expected will be "float16" in the test
            min: -0.3,
            max: 0.3,
            mean: 0.0,
            std: 0.2,
            hasNaN: false,
            hasInf: false
        )
    }

    // MARK: - Synthetic-event pair test

    /// Simulate a step-2 NaN anomaly:
    ///   • Push a denoiseStepComplete at stepIndex 2 whose noisePredStat has hasNaN == true.
    ///   • Push a numericalAnomaly(phase:kind:stepIndex:stat:) alongside it.
    ///   • Assert both events are recorded and carry consistent data.
    @Test func step2NaNAnomalyFiresBothEvents() async {
        let reporter = MockTelemetryReporter()
        let anomalyStat = Self.nanStat()

        // The step-2 denoiseStepComplete with a NaN noisePredStat.
        await reporter.capture(
            .denoiseStepComplete(
                variant: .textToImage,
                stepIndex: 2,
                totalSteps: 4,
                sigma: 0.6,
                timestep: 600.0,
                latentBeforeStat: Self.normalStat(),
                noisePredStat: anomalyStat,
                latentAfterStat: Self.normalStat(),
                kvCacheLayerCount: nil,
                kvCacheHit: nil,
                durationSeconds: 0.03
            )
        )

        // The side-channel anomaly event emitted alongside the step.
        await reporter.capture(
            .numericalAnomaly(
                phase: "denoiseStepComplete",
                kind: .nan,
                stepIndex: 2,
                stat: anomalyStat
            )
        )

        let events = await reporter.events()
        #expect(events.count == 2, "Expected 2 events (step + anomaly), got \(events.count)")

        // --- Validate denoiseStepComplete at step 2 carries NaN noisePredStat ---
        let stepEventsWithNaN: [(stepIndex: Int, noisePredStatHasNaN: Bool)] = events.compactMap { event in
            if case let .denoiseStepComplete(
                _, stepIndex, _, _, _, _, noisePredStat, _, _, _, _
            ) = event {
                return (stepIndex: stepIndex, noisePredStatHasNaN: noisePredStat.hasNaN)
            }
            return nil
        }
        #expect(stepEventsWithNaN.count == 1, "Expected exactly 1 denoiseStepComplete event")
        if let step = stepEventsWithNaN.first {
            #expect(step.stepIndex == 2, "Step index should be 2, got \(step.stepIndex)")
            #expect(step.noisePredStatHasNaN == true,
                    "noisePredStat.hasNaN should be true at step 2")
        }

        // --- Validate numericalAnomaly(.nan) fires at step 2 ---
        let anomalyEvents: [(phase: String, kind: Flux2TelemetryEvent.AnomalyKind, stepIndex: Int?)] =
            events.compactMap { event in
                if case let .numericalAnomaly(phase, kind, stepIndex, _) = event {
                    return (phase: phase, kind: kind, stepIndex: stepIndex)
                }
                return nil
            }
        #expect(anomalyEvents.count == 1, "Expected exactly 1 numericalAnomaly event")
        if let anomaly = anomalyEvents.first {
            #expect(anomaly.phase == "denoiseStepComplete",
                    "Anomaly phase should be 'denoiseStepComplete', got '\(anomaly.phase)'")
            #expect(anomaly.kind == .nan,
                    "Anomaly kind should be .nan, got \(anomaly.kind)")
            #expect(anomaly.stepIndex == 2,
                    "Anomaly stepIndex should be 2, got \(String(describing: anomaly.stepIndex))")
        }
    }

    // MARK: - Flux2AnomalyDetector direct tests

    /// Verify that Flux2AnomalyDetector.anomalies(in:) correctly identifies .nan.
    @Test func anomalyDetectorIdentifiesNaN() {
        let anomalies = Flux2AnomalyDetector.anomalies(in: Self.nanStat())
        #expect(anomalies.contains(.nan),
                "Expected .nan in anomalies for a NaN-flagged stat, got \(anomalies)")
    }

    /// Verify that Flux2AnomalyDetector.anomalies(in:) correctly identifies .inf.
    @Test func anomalyDetectorIdentifiesInf() {
        let anomalies = Flux2AnomalyDetector.anomalies(in: Self.infStat())
        #expect(anomalies.contains(.inf),
                "Expected .inf in anomalies for an Inf-flagged stat, got \(anomalies)")
    }

    /// Verify that Flux2AnomalyDetector.anomalies(in:) correctly identifies .outOfRange
    /// when max exceeds defaultOutOfRangeThreshold (1e6).
    @Test func anomalyDetectorIdentifiesOutOfRange() {
        let anomalies = Flux2AnomalyDetector.anomalies(in: Self.outOfRangeStat())
        #expect(anomalies.contains(.outOfRange),
                "Expected .outOfRange in anomalies for a stat with |max|>1e6, got \(anomalies)")
    }

    /// Verify that Flux2AnomalyDetector.anomalies(in:checkZeroLatent:) correctly identifies
    /// .zeroLatent when mean ≈ 0 and std ≈ 0 and checkZeroLatent is true.
    @Test func anomalyDetectorIdentifiesZeroLatent() {
        let anomalies = Flux2AnomalyDetector.anomalies(
            in: Self.zeroLatentStat(),
            checkZeroLatent: true
        )
        #expect(anomalies.contains(.zeroLatent),
                "Expected .zeroLatent in anomalies for a near-zero mean+std stat with checkZeroLatent: true, got \(anomalies)")
    }

    /// Verify that .zeroLatent is NOT reported when checkZeroLatent is false (the default).
    @Test func anomalyDetectorDoesNotReportZeroLatentWhenFlagIsFalse() {
        let anomalies = Flux2AnomalyDetector.anomalies(in: Self.zeroLatentStat())
        #expect(!anomalies.contains(.zeroLatent),
                ".zeroLatent should not fire when checkZeroLatent is false")
    }

    /// Verify that Flux2AnomalyDetector.anomalies(in:expectedDtype:) correctly identifies
    /// .dtypeUnexpected when the stat's dtype does not match the expected dtype.
    @Test func anomalyDetectorIdentifiesDTypeUnexpected() {
        // stat carries "float32", but we expect "float16"
        let anomalies = Flux2AnomalyDetector.anomalies(
            in: Self.dtypeMismatchStat(actual: "float32"),
            expectedDtype: "float16"
        )
        #expect(anomalies.contains(.dtypeUnexpected),
                "Expected .dtypeUnexpected when stat dtype ('float32') != expectedDtype ('float16'), got \(anomalies)")
    }

    /// Verify that a normal stat (no anomaly conditions) produces an empty anomalies array.
    @Test func anomalyDetectorProducesNoAnomaliesForNormalStat() {
        let anomalies = Flux2AnomalyDetector.anomalies(
            in: Self.normalStat(),
            checkZeroLatent: true,
            expectedDtype: "float16"
        )
        #expect(anomalies.isEmpty,
                "Expected no anomalies for a clean stat, got \(anomalies)")
    }
}
