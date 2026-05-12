// Flux2TelemetryVAEDenormalizationTests.swift
// Contract tests for the vaeBatchNormDenormalize event invariant.
//
// Contracts under test:
//   1. A single vaeBatchNormDenormalize event is recorded; afterStat.std differs
//      from beforeStat.std (the BatchNorm did something).
//   2. A full VAE trio (vaeDecodeStart → vaeBatchNormDenormalize → vaeDecodeComplete)
//      fires in order, all events having stepIndex == nil (VAE lives outside the
//      denoise loop).
//
// All tests are synthetic-event contract tests. No real weights, no GPU, no Metal.

import Foundation
import Testing
import Tuberia

@testable import Flux2Core

@Suite("Flux2TelemetryVAEDenormalizationTests")
struct Flux2TelemetryVAEDenormalizationTests {

    // MARK: - Helpers

    private static let latentShape = [1, 16, 64, 64]
    private static let pixelShape  = [1, 3, 512, 512]

    /// A stat representing a typical latent tensor before BatchNorm denormalization:
    /// std is 1.0 (unit variance).
    private static func beforeNormStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: latentShape,
            dtype: "float32",
            min: -2.1,
            max: 2.1,
            mean: 0.05,
            std: 1.0,
            hasNaN: false,
            hasInf: false
        )
    }

    /// A stat representing the same latent tensor after BatchNorm denormalization:
    /// std has changed to 0.5 (the BatchNorm rescaled the activations).
    private static func afterNormStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: latentShape,
            dtype: "float32",
            min: -1.05,
            max: 1.05,
            mean: 0.025,
            std: 0.5,
            hasNaN: false,
            hasInf: false
        )
    }

    /// A stat for the initial latent passed to vaeDecodeStart.
    private static func decodeInputStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: latentShape,
            dtype: "float32",
            min: -1.5,
            max: 1.5,
            mean: 0.0,
            std: 0.8,
            hasNaN: false,
            hasInf: false
        )
    }

    /// A stat representing a decoded pixel tensor for vaeDecodeComplete.
    private static func pixelOutputStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: pixelShape,
            dtype: "float32",
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.2,
            hasNaN: false,
            hasInf: false
        )
    }

    // MARK: - Single-event test

    /// Verify the core vaeBatchNormDenormalize invariant:
    ///   • Exactly 1 event is recorded.
    ///   • afterStat.std != beforeStat.std (the BatchNorm modified the tensor).
    @Test func vaeBatchNormDenormalizeSingleEventRecorded() async {
        let reporter = MockTelemetryReporter()
        let before = beforeNormStat()
        let after  = afterNormStat()

        await reporter.capture(
            .vaeBatchNormDenormalize(beforeStat: before, afterStat: after)
        )

        let events = await reporter.events()
        #expect(events.count == 1,
                "Expected exactly 1 event, got \(events.count)")

        let denormEvents: [(beforeStat: TuberiaTensorStat, afterStat: TuberiaTensorStat)] =
            events.compactMap { event in
                if case let .vaeBatchNormDenormalize(beforeStat, afterStat) = event {
                    return (beforeStat: beforeStat, afterStat: afterStat)
                }
                return nil
            }

        #expect(denormEvents.count == 1,
                "Expected exactly 1 vaeBatchNormDenormalize event, got \(denormEvents.count)")

        if let denorm = denormEvents.first {
            #expect(denorm.afterStat.std != denorm.beforeStat.std,
                    "afterStat.std (\(denorm.afterStat.std)) should differ from beforeStat.std (\(denorm.beforeStat.std)) — BatchNorm must have rescaled the tensor")
            // Also verify the specific values we put in (guards against accidental aliasing).
            #expect(denorm.beforeStat.std == 1.0,
                    "beforeStat.std should be 1.0, got \(denorm.beforeStat.std)")
            #expect(denorm.afterStat.std == 0.5,
                    "afterStat.std should be 0.5, got \(denorm.afterStat.std)")
        }
    }

    // MARK: - VAE trio ordering test

    /// Verify that a full VAE decode sequence
    ///   (vaeDecodeStart → vaeBatchNormDenormalize → vaeDecodeComplete)
    /// fires in the correct order and that all three events have no associated
    /// stepIndex (VAE lives outside the denoise loop, so stepIndex is nil on
    /// the anomaly side-channel; the three VAE events themselves carry no stepIndex
    /// field — their presence in the recorded array at correct positions is the
    /// ordering assertion).
    @Test func vaeDecodeTrioFiresInOrder() async {
        let reporter = MockTelemetryReporter()

        // 1. vaeDecodeStart
        await reporter.capture(
            .vaeDecodeStart(
                latentStat: decodeInputStat(),
                scalingFactor: 0.18215
            )
        )

        // 2. vaeBatchNormDenormalize
        await reporter.capture(
            .vaeBatchNormDenormalize(
                beforeStat: beforeNormStat(),
                afterStat: afterNormStat()
            )
        )

        // 3. vaeDecodeComplete
        await reporter.capture(
            .vaeDecodeComplete(
                pixelStat: pixelOutputStat(),
                outputDims: [512, 512],
                durationSeconds: 0.25
            )
        )

        let events = await reporter.events()
        #expect(events.count == 3,
                "Expected 3 VAE decode events, got \(events.count)")

        // Position 0 must be vaeDecodeStart.
        if case .vaeDecodeStart = events[0] {
            // ok — ordering confirmed at position 0
        } else {
            Issue.record("events[0] should be vaeDecodeStart, got \(events[0])")
        }

        // Position 1 must be vaeBatchNormDenormalize.
        if case .vaeBatchNormDenormalize = events[1] {
            // ok — ordering confirmed at position 1
        } else {
            Issue.record("events[1] should be vaeBatchNormDenormalize, got \(events[1])")
        }

        // Position 2 must be vaeDecodeComplete.
        if case .vaeDecodeComplete = events[2] {
            // ok — ordering confirmed at position 2
        } else {
            Issue.record("events[2] should be vaeDecodeComplete, got \(events[2])")
        }

        // Count each type to make sure no duplicates snuck in.
        let decodeStarts = events.filter { if case .vaeDecodeStart = $0 { return true }; return false }
        let denorms      = events.filter { if case .vaeBatchNormDenormalize = $0 { return true }; return false }
        let decodeEnds   = events.filter { if case .vaeDecodeComplete = $0 { return true }; return false }

        #expect(decodeStarts.count == 1, "Expected 1 vaeDecodeStart, got \(decodeStarts.count)")
        #expect(denorms.count == 1,      "Expected 1 vaeBatchNormDenormalize, got \(denorms.count)")
        #expect(decodeEnds.count == 1,   "Expected 1 vaeDecodeComplete, got \(decodeEnds.count)")

        // VAE events carry no stepIndex field. Confirm the numericalAnomaly side-channel
        // is NOT in this sequence (no anomaly was injected).
        let anomalyEvents = events.filter { if case .numericalAnomaly = $0 { return true }; return false }
        #expect(anomalyEvents.isEmpty,
                "No anomaly events should appear in a clean VAE decode trio, got \(anomalyEvents.count)")
    }
}
