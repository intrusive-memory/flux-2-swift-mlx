// Flux2TelemetryNoopOverheadTests.swift — Sortie 9 baseline overhead contract.
//
// Q10 mocked-transformer rig: the test does NOT run real MLX kernels. Instead,
// it simulates the per-step shape that the production denoise loop performs
// (see `Flux2Pipeline.swift:2079` — the textToImage loop) with a constant-time
// stand-in for the "transformer" call. The transformer stub performs a fixed
// amount of CPU spin work per step so the per-step wall-clock is dominated by
// "kernel" time, not by the loop scaffolding. That mirrors production, where
// a single transformer forward + scheduler.step + eval takes 10s of
// milliseconds and telemetry is a fraction of one percent.
//
// Three cohorts × 20 iterations × 20 steps:
//   - A: no telemetry installed                                  (baseline)
//   - B: NoopFlux2TelemetryReporter                              (guard passes; capture is a no-op)
//   - C: MockTelemetryReporter                                   (full event recording)
//
// Contract bounds (Q10, REQUIREMENTS §7/§8):
//   - Cohort B median ≤ Cohort A × 1.01  (+1% Noop bound)
//   - Cohort C median ≤ Cohort A × 1.05  (+5% Mock bound)
//
// On success the test prints
//     OVERHEAD_NOOP_PCT=<X> OVERHEAD_MOCK_PCT=<Y>
// to stdout. Sortie 10's PR description picks this line up.
//
// F8: `import TestHelpers` brings `MockTelemetryReporter` into scope.
// F10: every helper is a non-static instance method.

import Foundation
import Testing
import TestHelpers
import Tuberia
import os.lock

@testable import Flux2Core

@Suite("Flux2Telemetry overhead contract")
struct Flux2TelemetryNoopOverheadTests {

    // F10: NON-static helper. Builds a healthy synthetic stat (no anomalies)
    // matching the shape and dtype the production loop's
    // `denoiseStepComplete` carries.
    private func makeStat() -> TuberiaTensorStat {
        // Init signature (verified at SwiftTuberia v0.7.0 source): shape,
        // dtype, min, max, mean, std, hasNaN, hasInf. Numeric fields are
        // `Double` in Tuberia regardless of the underlying MLX dtype.
        TuberiaTensorStat(
            shape: [1, 16, 64, 64],
            dtype: "float16",
            min: -1.0,
            max: 1.0,
            mean: 0.0,
            std: 1.0,
            hasNaN: false,
            hasInf: false
        )
    }

    /// Run `iterations` rounds of `stepsPerIter` simulated denoise steps.
    /// Each step mirrors the production hot-path shape: one
    /// `currentTelemetry()` lock read, an `if let telemetry` guard, three
    /// stat samples plus a `denoiseStepComplete` capture, and the per-stat
    /// `numericalAnomaly` retrofit loop (no anomalies in the synthetic stat,
    /// so each retrofit body is no-op — same shape the production code
    /// pays on the healthy path).
    ///
    /// Returns the median per-iteration wall-clock duration in nanoseconds.
    /// Median is more robust than mean against GC / scheduler blips.
    private func runCohort(
        reporter: (any Flux2TelemetryReporter)?,
        iterations: Int,
        stepsPerIter: Int
    ) async -> Double {
        let seam = SeamUnderTest()
        seam.setTelemetry(reporter)

        // Pre-allocated synthetic stat. Allocation is excluded from timing
        // (constructed once before the warm-up + measurement loops).
        let stat = makeStat()

        // Warm-up: 5 iters of the hottest path so JIT, branch predictor, and
        // ARC paths settle before we start measuring.
        for _ in 0..<5 {
            for stepIdx in 0..<stepsPerIter {
                await seam.simulateStep(stat: stat, stepIdx: stepIdx, totalSteps: stepsPerIter)
            }
        }

        var elapsedNanos: [Double] = []
        elapsedNanos.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = ContinuousClock.now
            for stepIdx in 0..<stepsPerIter {
                await seam.simulateStep(stat: stat, stepIdx: stepIdx, totalSteps: stepsPerIter)
            }
            let end = ContinuousClock.now
            let delta = end - start
            // Components: (seconds: Int64, attoseconds: Int64). 1 ns = 1e9 as.
            let nanos = Double(delta.components.seconds) * 1_000_000_000.0
                + Double(delta.components.attoseconds) / 1_000_000_000.0
            elapsedNanos.append(nanos)
        }
        elapsedNanos.sort()
        return elapsedNanos[elapsedNanos.count / 2]
    }

    /// Mirror of `Flux2Pipeline`'s telemetry lock seam (`Sortie 2` pattern)
    /// plus a synthetic per-step emission matching the production
    /// `denoiseStepComplete` hot path. We use a mirror rather than
    /// instantiating `Flux2Pipeline` directly because the real pipeline
    /// requires loading models, which would dominate wall-clock time and
    /// defeat the purpose of measuring telemetry overhead.
    private final class SeamUnderTest: @unchecked Sendable {
        private let lock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(initialState: nil)

        // Sink for the simulated-transformer CPU spin. `nonisolated(unsafe)`
        // because we deliberately want a non-cancellable side effect the
        // compiler cannot elide — the actual value is meaningless.
        nonisolated(unsafe) static var spinSink: UInt64 = 0

        /// Constant-time CPU spin that stands in for the transformer
        /// forward + scheduler.step + eval cost per step. Tuned so the
        /// per-step total dwarfs telemetry overhead (which is on the order
        /// of hundreds of ns to a few μs per step), giving the +1% Noop
        /// and +5% Mock bounds room to be measured reliably.
        func simulateTransformerWork() {
            // ~80–120 μs of busy work on Apple Silicon. We write the result
            // back to a static so the compiler cannot prove this is dead.
            var acc: UInt64 = SeamUnderTest.spinSink &+ 1
            for i in 0..<20_000 {
                acc = (acc &* 6_364_136_223_846_793_005) &+ UInt64(i)
            }
            SeamUnderTest.spinSink = acc
        }

        func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?) {
            lock.withLock { $0 = reporter }
        }

        // Mirrors Flux2Pipeline.currentTelemetry().
        func currentTelemetry() -> (any Flux2TelemetryReporter)? {
            lock.withLock { $0 }
        }

        /// One step's worth of emission work, matching the production hot
        /// path body in `Flux2Pipeline.swift:2079`:
        ///   0. simulated "transformer" CPU work — a constant-time spin
        ///      that approximates a real forward pass + scheduler step
        ///      well enough to be the dominant cost per step.
        ///   1. acquire telemetry pointer exactly once via `currentTelemetry()`
        ///   2. if-let guard: three stat samples (latentBefore, noisePred,
        ///      latentAfter) — already cached as the synthetic stat
        ///   3. emit `.denoiseStepComplete`
        ///   4. anomaly-retrofit loop over each of the 3 stats
        ///
        /// The transformer "kernel" is replaced by a deterministic CPU spin
        /// so the cohort medians can resolve telemetry overhead as a small
        /// fraction of the per-step total — same proportions as in
        /// production, just compressed in wall-clock terms.
        func simulateStep(stat: TuberiaTensorStat, stepIdx: Int, totalSteps: Int) async {
            simulateTransformerWork()
            let telemetry = currentTelemetry()
            if let telemetry {
                let latentBefore = stat
                let noisePred = stat
                let latentAfter = stat
                // sigma + timestep are `Float` in the case payload.
                let sigma = Float(1.0) - Float(stepIdx) * Float(0.05)
                let timestep = sigma
                await telemetry.capture(
                    .denoiseStepComplete(
                        variant: .textToImage,
                        stepIndex: stepIdx,
                        totalSteps: totalSteps,
                        sigma: sigma,
                        timestep: timestep,
                        latentBeforeStat: latentBefore,
                        noisePredStat: noisePred,
                        latentAfterStat: latentAfter,
                        kvCacheLayerCount: nil,
                        kvCacheHit: nil,
                        durationSeconds: 0.001
                    ))
                for kind in Flux2AnomalyDetector.anomalies(in: latentBefore, checkZeroLatent: true) {
                    await telemetry.capture(
                        .numericalAnomaly(
                            phase: "denoiseStepComplete",
                            kind: kind,
                            stepIndex: stepIdx,
                            stat: latentBefore
                        ))
                }
                for kind in Flux2AnomalyDetector.anomalies(in: noisePred, checkZeroLatent: true) {
                    await telemetry.capture(
                        .numericalAnomaly(
                            phase: "denoiseStepComplete",
                            kind: kind,
                            stepIndex: stepIdx,
                            stat: noisePred
                        ))
                }
                for kind in Flux2AnomalyDetector.anomalies(in: latentAfter, checkZeroLatent: true) {
                    await telemetry.capture(
                        .numericalAnomaly(
                            phase: "denoiseStepComplete",
                            kind: kind,
                            stepIndex: stepIdx,
                            stat: latentAfter
                        ))
                }
            }
        }
    }

    @Test("Noop overhead <= +1% over baseline; Mock overhead <= +5% over baseline")
    func testOverhead() async {
        // Q10: 3 cohorts × 20 iterations × 20 steps.
        let iterations = 20
        let stepsPerIter = 20

        // Cohort A: no telemetry — baseline (the `if let telemetry` guard
        // short-circuits, so the loop pays only the `currentTelemetry()`
        // lock read per step).
        let medianA = await runCohort(
            reporter: nil,
            iterations: iterations,
            stepsPerIter: stepsPerIter
        )
        // Cohort B: Noop reporter — guard branch is taken; samples + anomaly
        // detector + capture run, but `capture` itself is a no-op.
        let medianB = await runCohort(
            reporter: NoopFlux2TelemetryReporter(),
            iterations: iterations,
            stepsPerIter: stepsPerIter
        )
        // Cohort C: Mock reporter — full event recording (actor append).
        let medianC = await runCohort(
            reporter: MockTelemetryReporter(),
            iterations: iterations,
            stepsPerIter: stepsPerIter
        )

        let noopPct = (medianB - medianA) / medianA * 100.0
        let mockPct = (medianC - medianA) / medianA * 100.0
        let noopBound = medianA * 1.01  // +1%
        let mockBound = medianA * 1.05  // +5%

        // Sortie 10 reads this line out of the make-test stdout for the PR.
        print("OVERHEAD_NOOP_PCT=\(noopPct) OVERHEAD_MOCK_PCT=\(mockPct)")
        print("medianA=\(medianA) medianB=\(medianB) medianC=\(medianC)")

        #expect(
            medianB <= noopBound,
            "Noop overhead \(noopPct)% exceeds +1% bound (medianA=\(medianA) medianB=\(medianB))"
        )
        #expect(
            medianC <= mockBound,
            "Mock overhead \(mockPct)% exceeds +5% bound (medianA=\(medianA) medianC=\(medianC))"
        )
    }
}
