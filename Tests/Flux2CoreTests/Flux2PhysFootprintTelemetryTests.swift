// Flux2PhysFootprintTelemetryTests.swift
// Sortie A6 — per-phase phys_footprint telemetry.
//
// Asserts that each of the four phase-boundary events on the
// Flux2TelemetryEvent seam carries a non-nil `physFootprint` value:
//   - weightLoadComplete
//   - textEncodeComplete
//   - denoiseLoopEnd
//   - vaeDecodeComplete
//
// Strategy (CI-safe, no GPU / no model weights): the pipeline samples
// `Flux2MemoryFootprint.current()` at each boundary emit site. These tests
// assemble each event exactly as the pipeline does — passing
// `Flux2MemoryFootprint.current()` as the `physFootprint` argument — route it
// through MockFlux2TelemetryReporter, then destructure the captured event and
// assert the phys_footprint field is non-nil. This mirrors the Strategy-B
// pattern already used in Flux2TelemetryAnomalyTests.

import TestHelpers
import Testing
import Tuberia

@testable import Flux2Core

// MARK: - Helper for a minimal TuberiaTensorStat (no MLXArray needed)

private func makeStat() -> TuberiaTensorStat {
  TuberiaTensorStat(
    shape: [1, 4, 4, 4],
    dtype: "float32",
    min: -0.8,
    max: 0.8,
    mean: 0.01,
    std: 0.3,
    hasNaN: false,
    hasInf: false
  )
}

@Suite("Flux2 Telemetry — per-phase phys_footprint")
struct Flux2PhysFootprintTelemetryTests {

  // MARK: - The helper itself

  /// On macOS/arm64 CI hosts the `task_info(TASK_VM_INFO)` sample must
  /// succeed, so the helper returns a non-nil, strictly-positive footprint.
  @Test func physFootprintHelper_returnsNonNilPositiveValue() {
    let footprint = Flux2MemoryFootprint.current()
    #expect(footprint != nil, "Flux2MemoryFootprint.current() must return a value on the test host")
    if let footprint {
      #expect(footprint > 0, "phys_footprint must be strictly positive; got \(footprint)")
    }
  }

  // MARK: - weightLoadComplete

  @Test func weightLoadComplete_carriesNonNilPhysFootprint() async {
    let reporter = MockFlux2TelemetryReporter()
    await reporter.capture(
      .weightLoadComplete(
        component: .transformer,
        paramCount: 1_000,
        durationSeconds: 0.5,
        physFootprint: Flux2MemoryFootprint.current()))

    let events = await reporter.snapshot()
    guard case .weightLoadComplete(_, _, _, let physFootprint) = events.first else {
      Issue.record("Expected weightLoadComplete; got \(String(describing: events.first))")
      return
    }
    #expect(physFootprint != nil, "weightLoadComplete must carry a non-nil phys_footprint")
  }

  // MARK: - textEncodeComplete

  @Test func textEncodeComplete_carriesNonNilPhysFootprint() async {
    let reporter = MockFlux2TelemetryReporter()
    await reporter.capture(
      .textEncodeComplete(
        encoderName: "test-encoder",
        finalPromptLength: 77,
        embeddingStat: makeStat(),
        durationSeconds: 0.25,
        physFootprint: Flux2MemoryFootprint.current()))

    let events = await reporter.snapshot()
    guard case .textEncodeComplete(_, _, _, _, let physFootprint) = events.first else {
      Issue.record("Expected textEncodeComplete; got \(String(describing: events.first))")
      return
    }
    #expect(physFootprint != nil, "textEncodeComplete must carry a non-nil phys_footprint")
  }

  // MARK: - denoiseLoopEnd

  @Test func denoiseLoopEnd_carriesNonNilPhysFootprint() async {
    let reporter = MockFlux2TelemetryReporter()
    await reporter.capture(
      .denoiseLoopEnd(
        variant: .textToImage,
        totalSteps: 4,
        completedSteps: 4,
        finalLatentStat: makeStat(),
        durationSeconds: 12.0,
        physFootprint: Flux2MemoryFootprint.current()))

    let events = await reporter.snapshot()
    guard case .denoiseLoopEnd(_, _, _, _, _, let physFootprint) = events.first else {
      Issue.record("Expected denoiseLoopEnd; got \(String(describing: events.first))")
      return
    }
    #expect(physFootprint != nil, "denoiseLoopEnd must carry a non-nil phys_footprint")
  }

  // MARK: - vaeDecodeComplete

  @Test func vaeDecodeComplete_carriesNonNilPhysFootprint() async {
    let reporter = MockFlux2TelemetryReporter()
    await reporter.capture(
      .vaeDecodeComplete(
        pixelStat: makeStat(),
        outputDims: [1, 3, 768, 768],
        durationSeconds: 1.5,
        physFootprint: Flux2MemoryFootprint.current()))

    let events = await reporter.snapshot()
    guard case .vaeDecodeComplete(_, _, _, let physFootprint) = events.first else {
      Issue.record("Expected vaeDecodeComplete; got \(String(describing: events.first))")
      return
    }
    #expect(physFootprint != nil, "vaeDecodeComplete must carry a non-nil phys_footprint")
  }
}
