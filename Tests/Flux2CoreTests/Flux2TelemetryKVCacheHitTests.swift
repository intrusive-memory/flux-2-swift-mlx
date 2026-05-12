// Flux2TelemetryKVCacheHitTests.swift
// Contract tests for kvCacheHit / kvCacheLayerCount policy on denoiseStepComplete.
//
// Q7 policy under test:
//   • Klein9B-KV run: step 0 uses .imageToImageKVExtractStep0 with kvCacheHit == nil;
//     steps 1–N use .imageToImageKVCached with kvCacheHit == true and a non-nil
//     kvCacheLayerCount equal to the extract-step's kvCacheLayerCount.
//   • Text-to-image run: ALL steps have kvCacheLayerCount == nil and kvCacheHit == nil
//     (no KV cache participation at all).
//
// All tests are synthetic-event contract tests. No real weights, no GPU, no Metal.

import Foundation
import Testing
import Tuberia

@testable import Flux2Core

@Suite("Flux2TelemetryKVCacheHitTests")
struct Flux2TelemetryKVCacheHitTests {

    // MARK: - Helpers

    /// A plain stat suitable for any synthetic step that isn't specifically under test.
    private static func genericStat() -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 16, 8, 8],
            dtype: "float16",
            min: -0.5,
            max: 0.5,
            mean: 0.0,
            std: 0.25,
            hasNaN: false,
            hasInf: false
        )
    }

    /// Push a single `denoiseStepComplete` event for a Klein9B-KV run through `reporter`.
    ///
    /// - Parameters:
    ///   - stepIndex: 0–3
    ///   - variant: `.imageToImageKVExtractStep0` for step 0; `.imageToImageKVCached` for 1–3
    ///   - kvCacheLayerCount: 38 for all four steps
    ///   - kvCacheHit: `nil` for step 0, `true` for steps 1–3
    private static func captureKlein9BKVStep(
        reporter: MockTelemetryReporter,
        stepIndex: Int,
        variant: Flux2TelemetryEvent.DenoiseVariant,
        kvCacheLayerCount: Int?,
        kvCacheHit: Bool?
    ) async {
        await reporter.capture(
            .denoiseStepComplete(
                variant: variant,
                stepIndex: stepIndex,
                totalSteps: 4,
                sigma: Float(1.0 - Double(stepIndex) * 0.2),
                timestep: Float(1000 - stepIndex * 200),
                latentBeforeStat: genericStat(),
                noisePredStat: genericStat(),
                latentAfterStat: genericStat(),
                kvCacheLayerCount: kvCacheLayerCount,
                kvCacheHit: kvCacheHit,
                durationSeconds: 0.01 * Double(stepIndex + 1)
            )
        )
    }

    // MARK: - Klein9B-KV cohort tests

    /// Simulate a Klein9B-KV 4-step run and verify all KV-cache policy invariants:
    ///   • 4 denoiseStepComplete events total
    ///   • step 0: variant == .imageToImageKVExtractStep0, kvCacheHit == nil
    ///   • steps 1–3: variant == .imageToImageKVCached, kvCacheHit == true
    ///   • all four steps share the same kvCacheLayerCount (38)
    @Test func klein9BKVRunProducesCorrectKVCachePolicy() async {
        let reporter = MockTelemetryReporter()
        let kvLayerCount = 38

        // Step 0 — extract
        await captureKlein9BKVStep(
            reporter: reporter,
            stepIndex: 0,
            variant: .imageToImageKVExtractStep0,
            kvCacheLayerCount: kvLayerCount,
            kvCacheHit: nil
        )

        // Steps 1–3 — cached
        for stepIndex in 1...3 {
            await captureKlein9BKVStep(
                reporter: reporter,
                stepIndex: stepIndex,
                variant: .imageToImageKVCached,
                kvCacheLayerCount: kvLayerCount,
                kvCacheHit: true
            )
        }

        let events = await reporter.events()

        // Filter to denoiseStepComplete only.
        let stepEvents: [(variant: Flux2TelemetryEvent.DenoiseVariant,
                          stepIndex: Int,
                          kvCacheLayerCount: Int?,
                          kvCacheHit: Bool?)] = events.compactMap { event in
            if case let .denoiseStepComplete(
                variant, stepIndex, _, _, _, _, _, _, kvCacheLayerCount, kvCacheHit, _
            ) = event {
                return (variant: variant,
                        stepIndex: stepIndex,
                        kvCacheLayerCount: kvCacheLayerCount,
                        kvCacheHit: kvCacheHit)
            }
            return nil
        }

        // 4 total step events.
        #expect(stepEvents.count == 4,
                "Expected 4 denoiseStepComplete events, got \(stepEvents.count)")

        // Step 0 contract.
        let step0 = stepEvents[0]
        #expect(step0.variant == .imageToImageKVExtractStep0,
                "Step 0 variant should be imageToImageKVExtractStep0, got \(step0.variant)")
        #expect(step0.kvCacheHit == nil,
                "Step 0 kvCacheHit should be nil (extract step does not read cache)")
        #expect(step0.kvCacheLayerCount == 38,
                "Step 0 kvCacheLayerCount should be 38, got \(String(describing: step0.kvCacheLayerCount))")

        // Steps 1–3 contract.
        for idx in 1...3 {
            let step = stepEvents[idx]
            #expect(step.variant == .imageToImageKVCached,
                    "Step \(idx) variant should be imageToImageKVCached, got \(step.variant)")
            #expect(step.kvCacheHit == true,
                    "Step \(idx) kvCacheHit should be true, got \(String(describing: step.kvCacheHit))")
            #expect(step.kvCacheLayerCount == 38,
                    "Step \(idx) kvCacheLayerCount should be 38, got \(String(describing: step.kvCacheLayerCount))")
        }

        // All four share the same kvCacheLayerCount.
        let layerCounts = stepEvents.map { $0.kvCacheLayerCount }
        let allMatchExpectedLayerCount = layerCounts.allSatisfy { $0 == kvLayerCount }
        #expect(allMatchExpectedLayerCount,
                "All steps should share kvCacheLayerCount=\(kvLayerCount), got \(layerCounts)")
    }

    // MARK: - textToImage cohort tests

    /// Simulate a standard 4-step textToImage run and assert that no KV-cache
    /// fields are populated: every step must have kvCacheLayerCount == nil and
    /// kvCacheHit == nil.
    @Test func textToImageRunHasNoKVCacheFields() async {
        let reporter = MockTelemetryReporter()

        for stepIndex in 0...3 {
            await reporter.capture(
                .denoiseStepComplete(
                    variant: .textToImage,
                    stepIndex: stepIndex,
                    totalSteps: 4,
                    sigma: Float(1.0 - Double(stepIndex) * 0.2),
                    timestep: Float(1000 - stepIndex * 200),
                    latentBeforeStat: genericStat(),
                    noisePredStat: genericStat(),
                    latentAfterStat: genericStat(),
                    kvCacheLayerCount: nil,
                    kvCacheHit: nil,
                    durationSeconds: 0.01 * Double(stepIndex + 1)
                )
            )
        }

        let events = await reporter.events()

        let stepEvents: [(stepIndex: Int,
                          kvCacheLayerCount: Int?,
                          kvCacheHit: Bool?)] = events.compactMap { event in
            if case let .denoiseStepComplete(
                _, stepIndex, _, _, _, _, _, _, kvCacheLayerCount, kvCacheHit, _
            ) = event {
                return (stepIndex: stepIndex,
                        kvCacheLayerCount: kvCacheLayerCount,
                        kvCacheHit: kvCacheHit)
            }
            return nil
        }

        #expect(stepEvents.count == 4,
                "Expected 4 denoiseStepComplete events for textToImage run, got \(stepEvents.count)")

        for step in stepEvents {
            #expect(step.kvCacheLayerCount == nil,
                    "textToImage step \(step.stepIndex) kvCacheLayerCount should be nil")
            #expect(step.kvCacheHit == nil,
                    "textToImage step \(step.stepIndex) kvCacheHit should be nil")
        }
    }
}
