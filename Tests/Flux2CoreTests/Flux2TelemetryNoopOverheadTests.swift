// Flux2TelemetryNoopOverheadTests.swift
// Sortie B16 — final closer.
//
// Asserts that the boundary emit path costs near-zero wall-clock time when the
// reporter does nothing. The cross-library §11 convention's promise — "the
// default cost of telemetry stays near zero so it ships enabled by default" —
// is only credible if a no-op reporter doesn't measurably slow a generation
// vs. a nil reporter.
//
// ## Harness shape (and why it isn't a real T2I generation)
//
// A real T2I generation requires GPU + downloaded model weights. Neither is
// available in CI-safe tests. The harness is therefore a synthetic loop that
// exercises the same `setTelemetry` → `currentTelemetry()` → `capture(...)`
// chain that a real generation traverses for each boundary event.
//
// `Flux2Pipeline.dispose()` is the smallest method that traverses that chain
// end-to-end: it acquires the `OSAllocatedUnfairLock`, reads the reporter,
// and awaits `capture(.pipelineDispose)`. A tight loop of `dispose()` calls
// is therefore a faithful proxy for the per-generation emit cost (a clean
// T2I generation fires ~6–10 boundary events: 1 pipelineInit, 3 weightLoad,
// 1 textEncode, 1 schedulerConfigured, 1 denoiseLoopStart/End pair,
// 1 vaeDecode, plus any numericalAnomaly side-channels).
//
// ## Q4 tolerance (carry-over from iteration 02)
//
// The plan's strict ±2% bound is unachievable on macos-26 CI runners — prior
// iterations observed >±5% variance on similar timing tests. The Q4 entry in
// EXECUTION_PLAN.md authorizes widening to ±10% as the documented hard-fail
// threshold. The observed ratio is printed so future iterations can tighten
// the bound if hardware stabilizes.

import Flux2Core
import Testing
import Foundation

@Suite("Flux2 Telemetry Noop Overhead")
struct Flux2TelemetryNoopOverheadTests {

    /// Compares wall-clock medians of two synthetic harnesses:
    ///   1. Pipeline with `setTelemetry(nil)` — the chain short-circuits at
    ///      the optional unwrap; no `capture(...)` is invoked.
    ///   2. Pipeline with `setTelemetry(NoopFlux2TelemetryReporter())` — the
    ///      chain reaches a `capture(...)` whose body is empty.
    ///
    /// The harness calls `dispose()` (single `.pipelineDispose` emit per call)
    /// `emitsPerIteration` times per measurement, and takes `iterations`
    /// independent measurements per branch. Medians are then compared.
    ///
    /// Hard-fail at ±10% (Q4-tuned bound). The observed ratio is printed so
    /// the value is visible in test output; tighten in a future iteration if
    /// CI hardware variance drops below ±5%.
    @Test func noopReporter_overheadWithinTolerance() async throws {
        let pipeline1 = Flux2Pipeline()  // nil reporter
        let pipeline2 = Flux2Pipeline()
        pipeline1.setTelemetry(nil)
        pipeline2.setTelemetry(NoopFlux2TelemetryReporter())

        // Let the detached pipelineInit Tasks settle before timing begins;
        // otherwise the first iteration's measurements include init Task
        // scheduling cost unrelated to the emit path.
        try await Task.sleep(for: .milliseconds(100))

        // Warm-up: prime the lock acquisition path on both pipelines so the
        // first measured iteration isn't dominated by cold-cache behavior.
        for _ in 0..<5 {
            await pipeline1.dispose()
            await pipeline2.dispose()
        }

        let iterations = 20
        let emitsPerIteration = 10  // proxy for boundary-event count per generation

        var nilDurations: [Double] = []
        var noopDurations: [Double] = []

        for _ in 0..<iterations {
            let nilStart = Date()
            for _ in 0..<emitsPerIteration { await pipeline1.dispose() }
            nilDurations.append(Date().timeIntervalSince(nilStart))

            let noopStart = Date()
            for _ in 0..<emitsPerIteration { await pipeline2.dispose() }
            noopDurations.append(Date().timeIntervalSince(noopStart))
        }

        let nilMedian = Self.median(of: nilDurations)
        let noopMedian = Self.median(of: noopDurations)

        // Guard against degenerate medians on extremely fast runs that round
        // to zero — fall back to comparing means in that case so we don't
        // divide by zero.
        let referenceNil = nilMedian > 0 ? nilMedian : Self.mean(of: nilDurations)
        let referenceNoop = noopMedian > 0 ? noopMedian : Self.mean(of: noopDurations)

        // If both are still effectively zero, the emit path is faster than
        // our measurement resolution — that's a pass, not a fail.
        guard referenceNil > 0 else {
            print("noop overhead: nil-reporter median is below measurement resolution (\(nilMedian)s); skipping ratio assertion")
            return
        }

        let ratio = referenceNoop / referenceNil
        let observedDelta = abs(ratio - 1.0)

        // Q4: ±5% is the documented target; ±10% is the hard-fail. Surface
        // the value either way so iteration audits can track drift.
        print(
            "noop overhead observed: nilMedian=\(nilMedian)s, noopMedian=\(noopMedian)s, "
                + "ratio=\(ratio), delta=\(observedDelta)"
        )
        if observedDelta > 0.05 {
            print("noop overhead ratio observed delta: \(observedDelta) (above ±5% target; under ±10% hard-fail)")
        }

        let hardBound = 0.10  // ±10% — Q4 hard-fail threshold
        #expect(
            observedDelta < hardBound,
            "noop overhead exceeded ±10% hard-fail (ratio=\(ratio), delta=\(observedDelta))"
        )

        // Tear-down housekeeping so the post-test snapshot is consistent.
        await pipeline1.dispose()
        await pipeline2.dispose()
    }

    // MARK: - statistics helpers (non-static avoids Swift-6 strict-mode call-site
    // qualifier churn; tests reach them as `Self.median(...)`).

    private static func median(of values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let n = sorted.count
        if n % 2 == 0 { return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
        return sorted[n / 2]
    }

    private static func mean(of values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }
}
