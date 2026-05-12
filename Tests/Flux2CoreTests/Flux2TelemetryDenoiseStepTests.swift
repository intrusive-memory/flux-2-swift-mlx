// Flux2TelemetryDenoiseStepTests.swift — Sortie 7a contract test for the
// denoise loop's telemetry shape (loopStart → N×stepComplete → loopEnd).
//
// Option A "pure contract" test: we construct synthetic events directly and
// push them through `MockTelemetryReporter`. No real model, no real MLX
// kernels — only the event-shape invariants the host adapter (Vinetas) relies
// on.
//
// F8: `import TestHelpers` brings `MockTelemetryReporter` into scope.
// F10: every helper is a non-static instance method.

import Foundation
import Testing
import TestHelpers
import Tuberia

@testable import Flux2Core

@Suite("Flux2TelemetryDenoiseStep contract")
struct Flux2TelemetryDenoiseStepTests {

    // F10: NON-static helper. The whole-tensor stat init is `Codable` /
    // `Equatable`, so chained-latent assertions can compare via `==` directly
    // if we want — we do scalar comparisons here for resilience to internal
    // field changes.
    private func makeStat(mean: Double = 0.0, std: Double = 1.0) -> TuberiaTensorStat {
        // Init signature (verified at SwiftTuberia v0.7.0 source): shape,
        // dtype, min, max, mean, std, hasNaN, hasInf.
        TuberiaTensorStat(
            shape: [1, 2, 16, 16],
            dtype: "float16",
            min: mean - 2 * std,
            max: mean + 2 * std,
            mean: mean,
            std: std,
            hasNaN: false,
            hasInf: false
        )
    }

    @Test("loop emits start + 4 step-complete + end with monotone stepIndex and chained latents")
    func testDenoiseLoopShape() async {
        let mock = MockTelemetryReporter()
        let totalSteps = 4
        let initialStat = makeStat()

        await mock.capture(
            .denoiseLoopStart(
                variant: .textToImage,
                totalSteps: totalSteps,
                latentShape: [1, 16, 32, 32],
                latentDtype: "float16",
                initialLatentStat: initialStat
            )
        )

        var previousAfter = initialStat
        for stepIdx in 0..<totalSteps {
            let before = previousAfter
            let noisePred = makeStat(mean: 0.1)
            let after = makeStat(mean: 0.2 + Double(stepIdx) * 0.1)
            await mock.capture(
                .denoiseStepComplete(
                    variant: .textToImage,
                    stepIndex: stepIdx,
                    totalSteps: totalSteps,
                    sigma: 1.0 - Float(stepIdx) * 0.25,
                    timestep: 1000.0 - Float(stepIdx) * 250.0,
                    latentBeforeStat: before,
                    noisePredStat: noisePred,
                    latentAfterStat: after,
                    kvCacheLayerCount: nil,
                    kvCacheHit: nil,
                    durationSeconds: 0.1
                )
            )
            previousAfter = after
        }

        await mock.capture(
            .denoiseLoopEnd(
                variant: .textToImage,
                totalSteps: totalSteps,
                completedSteps: totalSteps,
                finalLatentStat: previousAfter,
                durationSeconds: 0.4
            )
        )

        let events = await mock.events()
        #expect(events.count == 6)
        guard events.count == 6 else {
            Issue.record("expected 6 events (loopStart + 4×stepComplete + loopEnd), got \(events.count)")
            return
        }

        // First event must be `.denoiseLoopStart`.
        if case .denoiseLoopStart = events[0] {
            // ok
        } else {
            Issue.record("expected first event to be .denoiseLoopStart, got \(events[0])")
        }

        // Last event must be `.denoiseLoopEnd`.
        if case .denoiseLoopEnd = events[5] {
            // ok
        } else {
            Issue.record("expected last event to be .denoiseLoopEnd, got \(events[5])")
        }

        // Middle 4 events are `.denoiseStepComplete` with monotone stepIndex
        // 0..<4 and the latent-chaining invariant: stepComplete[N].after equals
        // stepComplete[N+1].before.
        var lastAfter: TuberiaTensorStat? = nil
        for offset in 0..<totalSteps {
            let event = events[1 + offset]
            guard
                case let .denoiseStepComplete(
                    variant,
                    stepIndex,
                    eventTotalSteps,
                    _,
                    _,
                    before,
                    _,
                    after,
                    kvLayerCount,
                    kvHit,
                    _
                ) = event
            else {
                Issue.record("expected .denoiseStepComplete at index \(1 + offset), got \(event)")
                continue
            }
            #expect(variant == .textToImage)
            #expect(stepIndex == offset)
            #expect(eventTotalSteps == totalSteps)
            #expect(kvLayerCount == nil)
            #expect(kvHit == nil)

            if let prevAfter = lastAfter {
                // Latent chaining: this step's `before` is the previous step's
                // `after`. Compare scalar fields for resilience; `==` would also
                // work because `TuberiaTensorStat` is Equatable.
                #expect(prevAfter.mean == before.mean)
                #expect(prevAfter.std == before.std)
                #expect(prevAfter.min == before.min)
                #expect(prevAfter.max == before.max)
                #expect(prevAfter.shape == before.shape)
                #expect(prevAfter.dtype == before.dtype)
            }
            lastAfter = after
        }
    }
}
