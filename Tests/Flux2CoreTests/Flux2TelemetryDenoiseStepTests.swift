// Flux2TelemetryDenoiseStepTests.swift
// Contract tests for the denoiseLoopStart / denoiseStepComplete / denoiseLoopEnd
// event sequence. Tests use MockTelemetryReporter (Sortie 7a) and synthesise
// events directly — no real weights, no GPU, no metal required — so they are
// fully CI-safe.
//
// The contract being tested:
//   • A 4-step T2I run produces exactly:
//       1× denoiseLoopStart
//       4× denoiseStepComplete  (stepIndex 0, 1, 2, 3 — monotone)
//       1× denoiseLoopEnd
//   • For consecutive steps N and N+1, latentAfterStat at step N ==
//     latentBeforeStat at step N+1 (chaining invariant).

import Foundation
import Testing
import Tuberia

@testable import Flux2Core

@Suite("Flux2TelemetryDenoiseStepTests")
struct Flux2TelemetryDenoiseStepTests {

    // MARK: - Helpers

    /// Convenience: build a deterministic `TuberiaTensorStat` for step `n`.
    private static func stat(step n: Int) -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 16, 8, 8],
            dtype: "float16",
            min: Double(n) * 0.1,
            max: Double(n) * 0.1 + 1.0,
            mean: Double(n) * 0.1 + 0.5,
            std: 0.3,
            hasNaN: false,
            hasInf: false
        )
    }

    /// Synthesise events for a 4-step T2I run and push them through `reporter`.
    ///
    /// Uses the same stat-chaining invariant the real pipeline must satisfy:
    ///   step[n].latentAfterStat == step[n+1].latentBeforeStat
    private static func simulateFourStepRun(reporter: MockTelemetryReporter) async {
        let totalSteps = 4
        let initialLatentStat = stat(step: 0)

        // --- loop start ---
        await reporter.capture(
            .denoiseLoopStart(
                variant: .textToImage,
                totalSteps: totalSteps,
                latentShape: [1, 16, 8, 8],
                latentDtype: "float16",
                initialLatentStat: initialLatentStat
            )
        )

        // --- 4 steps ---
        // latentBeforeStat at step N is stat(step: N)
        // latentAfterStat  at step N is stat(step: N+1)
        // So step[N+1].latentBeforeStat == stat(step: N+1) == step[N].latentAfterStat. ✓
        for stepIndex in 0..<totalSteps {
            await reporter.capture(
                .denoiseStepComplete(
                    variant: .textToImage,
                    stepIndex: stepIndex,
                    totalSteps: totalSteps,
                    sigma: Float(1.0 - Double(stepIndex) * 0.2),
                    timestep: Float(1000 - stepIndex * 200),
                    latentBeforeStat: stat(step: stepIndex),
                    noisePredStat: TuberiaTensorStat(
                        shape: [1, 16, 8, 8],
                        dtype: "float16",
                        min: -0.5,
                        max: 0.5,
                        mean: 0.0,
                        std: 0.25,
                        hasNaN: false,
                        hasInf: false
                    ),
                    latentAfterStat: stat(step: stepIndex + 1),
                    kvCacheLayerCount: nil,
                    kvCacheHit: nil,
                    durationSeconds: 0.01 * Double(stepIndex + 1)
                )
            )
        }

        // --- loop end ---
        await reporter.capture(
            .denoiseLoopEnd(
                variant: .textToImage,
                totalSteps: totalSteps,
                completedSteps: totalSteps,
                finalLatentStat: stat(step: totalSteps),
                durationSeconds: 0.04
            )
        )
    }

    // MARK: - Tests

    /// Verify the overall event shape: 1× start, 4× step, 1× end.
    @Test func fourStepRunProducesCorrectEventShape() async {
        let reporter = MockTelemetryReporter()
        await simulateFourStepRun(reporter: reporter)

        let events = await reporter.events()

        // --- count starts ---
        let starts = events.filter {
            if case .denoiseLoopStart = $0 { return true }
            return false
        }
        #expect(starts.count == 1, "Expected exactly 1 denoiseLoopStart, got \(starts.count)")

        // --- count steps ---
        let steps = events.filter {
            if case .denoiseStepComplete = $0 { return true }
            return false
        }
        #expect(steps.count == 4, "Expected exactly 4 denoiseStepComplete, got \(steps.count)")

        // --- count ends ---
        let ends = events.filter {
            if case .denoiseLoopEnd = $0 { return true }
            return false
        }
        #expect(ends.count == 1, "Expected exactly 1 denoiseLoopEnd, got \(ends.count)")
    }

    /// Verify stepIndex values are monotone: 0, 1, 2, 3.
    @Test func stepIndicesAreMonotonicallyIncreasing() async {
        let reporter = MockTelemetryReporter()
        await simulateFourStepRun(reporter: reporter)

        let events = await reporter.events()
        let stepIndices: [Int] = events.compactMap { event -> Int? in
            if case let .denoiseStepComplete(_, stepIndex, _, _, _, _, _, _, _, _, _) = event {
                return stepIndex
            }
            return nil
        }

        #expect(stepIndices == [0, 1, 2, 3],
                "Expected step indices [0,1,2,3], got \(stepIndices)")
    }

    /// Verify latent chaining: latentAfterStat at step N == latentBeforeStat at step N+1.
    @Test func latentStatChainingIsConsistent() async {
        let reporter = MockTelemetryReporter()
        await simulateFourStepRun(reporter: reporter)

        let events = await reporter.events()

        // Extract (latentBeforeStat, latentAfterStat) from each step event, in order.
        let statPairs: [(before: TuberiaTensorStat, after: TuberiaTensorStat)] = events.compactMap { event in
            if case let .denoiseStepComplete(
                _, _, _, _, _, latentBeforeStat, _, latentAfterStat, _, _, _) = event
            {
                return (before: latentBeforeStat, after: latentAfterStat)
            }
            return nil
        }

        #expect(statPairs.count == 4,
                "Need exactly 4 step events to check chaining, got \(statPairs.count)")

        // For each consecutive pair (N, N+1): step[N].after == step[N+1].before.
        for n in 0..<(statPairs.count - 1) {
            let afterN  = statPairs[n].after
            let beforeN1 = statPairs[n + 1].before
            #expect(afterN == beforeN1,
                    "Chaining broken at N=\(n): step[\(n)].latentAfterStat (\(afterN)) != step[\(n+1)].latentBeforeStat (\(beforeN1))")
        }
    }

    /// Verify that the ordering of events is: start, step, step, step, step, end.
    @Test func eventOrderingIsCorrect() async {
        let reporter = MockTelemetryReporter()
        await simulateFourStepRun(reporter: reporter)

        let events = await reporter.events()
        #expect(events.count == 6,
                "Expected 6 total events (1 start + 4 steps + 1 end), got \(events.count)")

        // First event must be loopStart.
        if case .denoiseLoopStart = events[0] {
            // ok
        } else {
            Issue.record("First event should be denoiseLoopStart, got \(events[0])")
        }

        // Last event must be loopEnd.
        if case .denoiseLoopEnd = events[5] {
            // ok
        } else {
            Issue.record("Last event should be denoiseLoopEnd, got \(events[5])")
        }

        // Middle four must all be denoiseStepComplete.
        for i in 1...4 {
            if case .denoiseStepComplete = events[i] {
                // ok
            } else {
                Issue.record("Event at index \(i) should be denoiseStepComplete, got \(events[i])")
            }
        }
    }
}
