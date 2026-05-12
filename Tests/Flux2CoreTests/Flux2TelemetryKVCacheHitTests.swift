// Flux2TelemetryKVCacheHitTests.swift — Sortie 7b contract test for KV-cache
// hit policy in the denoiseStepComplete event surface.
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

@Suite("Flux2Telemetry KV cache hit policy")
struct Flux2TelemetryKVCacheHitTests {

    // F10: NON-static helper. Matches the init order from
    // Flux2TelemetryDenoiseStepTests: (shape, dtype, min, max, mean, std, hasNaN, hasInf).
    private func makeStat(mean: Double = 0.0) -> TuberiaTensorStat {
        TuberiaTensorStat(
            shape: [1, 2, 16, 16],
            dtype: "float16",
            min: mean - 2.0,
            max: mean + 2.0,
            mean: mean,
            std: 1.0,
            hasNaN: false,
            hasInf: false
        )
    }

    @Test("Klein9BKV cohort: step 0 is KVExtractStep0 with kvCacheHit nil; steps 1-3 are KVCached with kvCacheHit true; all share same kvCacheLayerCount")
    func testKlein9BKVCohort() async {
        let mock = MockTelemetryReporter()
        let layerCount = 38   // arbitrary positive
        let stat = makeStat()

        // Step 0: KVExtractStep0 — layerCount set, hit nil
        await mock.capture(.denoiseStepComplete(
            variant: .imageToImageKVExtractStep0,
            stepIndex: 0,
            totalSteps: 4,
            sigma: 1.0,
            timestep: 1000,
            latentBeforeStat: stat,
            noisePredStat: stat,
            latentAfterStat: stat,
            kvCacheLayerCount: layerCount,
            kvCacheHit: nil,
            durationSeconds: 0.1
        ))

        // Steps 1-3: KVCached — layerCount set, hit true
        for stepIdx in 1...3 {
            await mock.capture(.denoiseStepComplete(
                variant: .imageToImageKVCached,
                stepIndex: stepIdx,
                totalSteps: 4,
                sigma: 1.0 - Float(stepIdx) * 0.25,
                timestep: 1000.0 - Float(stepIdx) * 250.0,
                latentBeforeStat: stat,
                noisePredStat: stat,
                latentAfterStat: stat,
                kvCacheLayerCount: layerCount,
                kvCacheHit: true,
                durationSeconds: 0.1
            ))
        }

        let events = await mock.events()
        #expect(events.count == 4)

        for (idx, event) in events.enumerated() {
            guard
                case let .denoiseStepComplete(
                    variant,
                    stepIndex,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    kvCacheLayerCount,
                    kvCacheHit,
                    _
                ) = event
            else {
                Issue.record("expected denoiseStepComplete at index \(idx)")
                continue
            }
            #expect(stepIndex == idx)
            #expect(kvCacheLayerCount == layerCount)
            if idx == 0 {
                #expect(variant == .imageToImageKVExtractStep0)
                #expect(kvCacheHit == nil)
            } else {
                #expect(variant == .imageToImageKVCached)
                #expect(kvCacheHit == true)
            }
        }
    }

    @Test("textToImage cohort: 4 steps, all kvCacheLayerCount nil and kvCacheHit nil")
    func testTextToImageCohort() async {
        let mock = MockTelemetryReporter()
        let stat = makeStat()
        for stepIdx in 0..<4 {
            await mock.capture(.denoiseStepComplete(
                variant: .textToImage,
                stepIndex: stepIdx,
                totalSteps: 4,
                sigma: 1.0 - Float(stepIdx) * 0.25,
                timestep: 1000.0 - Float(stepIdx) * 250.0,
                latentBeforeStat: stat,
                noisePredStat: stat,
                latentAfterStat: stat,
                kvCacheLayerCount: nil,
                kvCacheHit: nil,
                durationSeconds: 0.1
            ))
        }

        let events = await mock.events()
        #expect(events.count == 4)
        for event in events {
            guard
                case let .denoiseStepComplete(
                    variant,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    kvCacheLayerCount,
                    kvCacheHit,
                    _
                ) = event
            else {
                Issue.record("expected denoiseStepComplete")
                continue
            }
            #expect(variant == .textToImage)
            #expect(kvCacheLayerCount == nil)
            #expect(kvCacheHit == nil)
        }
    }
}
