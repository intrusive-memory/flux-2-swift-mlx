// Flux2TelemetryVAEDenormalizationTests.swift — Sortie 7b contract test for
// the VAE denormalize + decode telemetry event shape.
//
// Option A "pure contract" test: we construct synthetic events directly and
// push them through `MockTelemetryReporter`. No real model, no real MLX
// kernels — only the event-shape invariants the host adapter (Vinetas) relies
// on.
//
// Tests:
//   1. `vaeBatchNormDenormalize` carries before/after stats with differing std
//      (the canonical SD-VAE denorm scale is 0.18215).
//   2. The decode triplet ordering invariant:
//      vaeDecodeStart → vaeBatchNormDenormalize → vaeDecodeComplete.
//
// F8: `import TestHelpers` brings `MockTelemetryReporter` into scope.
// F10: every helper is a non-static instance method.

import Foundation
import Testing
import TestHelpers
import Tuberia

@testable import Flux2Core

@Suite("Flux2Telemetry VAE denormalize + decode contract")
struct Flux2TelemetryVAEDenormalizationTests {

    // F10: NON-static helper. Matches the TuberiaTensorStat init order:
    // (shape, dtype, min, max, mean, std, hasNaN, hasInf).
    private func stat(std: Double) -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 16, 64, 64],
            dtype: "float16",
            min: -2.0 * std,
            max: 2.0 * std,
            mean: 0.0,
            std: std,
            hasNaN: false,
            hasInf: false
        )
    }

    @Test("vaeBatchNormDenormalize event carries before/after stats with std change")
    func testDenormStatChange() async {
        let mock = MockTelemetryReporter()
        let before = stat(std: 1.0)
        let after  = stat(std: 0.18215)   // canonical SD-VAE denorm scale
        await mock.capture(.vaeBatchNormDenormalize(beforeStat: before, afterStat: after))

        let events = await mock.events()
        #expect(events.count == 1)
        if case let .vaeBatchNormDenormalize(beforeStat, afterStat) = events[0] {
            #expect(afterStat.std != beforeStat.std)
        } else {
            Issue.record("expected vaeBatchNormDenormalize, got \(events[0])")
        }
    }

    @Test("VAE-decode triplet ordering: vaeDecodeStart → vaeBatchNormDenormalize → vaeDecodeComplete")
    func testTripletOrdering() async {
        let mock = MockTelemetryReporter()
        let latent = stat(std: 1.0)
        let denormBefore = stat(std: 1.0)
        let denormAfter  = stat(std: 0.18215)
        let pixels = stat(std: 0.5)

        await mock.capture(.vaeDecodeStart(latentStat: latent, scalingFactor: 0.3611))
        await mock.capture(.vaeBatchNormDenormalize(beforeStat: denormBefore, afterStat: denormAfter))
        await mock.capture(.vaeDecodeComplete(
            pixelStat: pixels,
            outputDims: [1, 3, 1024, 1024],
            durationSeconds: 0.5
        ))

        let events = await mock.events()
        #expect(events.count == 3)

        if case .vaeDecodeStart = events[0] {
            // ok
        } else {
            Issue.record("expected vaeDecodeStart first, got \(events[0])")
        }

        if case .vaeBatchNormDenormalize = events[1] {
            // ok
        } else {
            Issue.record("expected vaeBatchNormDenormalize second, got \(events[1])")
        }

        if case .vaeDecodeComplete = events[2] {
            // ok
        } else {
            Issue.record("expected vaeDecodeComplete third, got \(events[2])")
        }
    }
}
