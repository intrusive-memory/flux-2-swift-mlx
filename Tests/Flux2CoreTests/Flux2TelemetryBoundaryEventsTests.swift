// Flux2TelemetryBoundaryEventsTests.swift
// Pattern-establishing test sortie for boundary-event telemetry (B12).
// Uses Swift Testing — consistent with Flux2CoreTests.swift.

import Flux2Core
import TestHelpers
import Testing

@Suite("Flux2 Telemetry Boundary Events")
struct Flux2TelemetryBoundaryEventsTests {

  // `pipelineInit` delivery is covered deterministically by
  // `Flux2ProcessWideTelemetryTests.processWideReporter_receivesInit_whenNoInstanceReporterSet`.
  // An instance reporter installed via `setTelemetry` after `init` races the
  // fire-and-forget init Task by design, so it cannot assert on `pipelineInit`.

  // MARK: - Detaching reporter silences subsequent events

  /// After setTelemetry(nil), calling dispose() must not deliver a
  /// pipelineDispose event to the previously installed reporter.
  @Test func boundaryEvents_detachReporter_silencesDispose() async throws {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)

    // Detach before dispose.
    pipeline.setTelemetry(nil)

    await pipeline.dispose()

    let events = await reporter.snapshot()
    let hasDispose = events.contains {
      if case .pipelineDispose = $0 { return true }
      return false
    }
    #expect(!hasDispose, "pipelineDispose must NOT fire after reporter is detached (nil)")
  }

  // MARK: - Double dispose emits twice

  /// Calling dispose() twice should fire pipelineDispose twice.
  @Test func boundaryEvents_doubleDispose_firesTwice() async throws {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)

    await pipeline.dispose()
    await pipeline.dispose()

    let events = await reporter.snapshot()
    let disposeCount = events.filter {
      if case .pipelineDispose = $0 { return true }
      return false
    }.count
    #expect(
      disposeCount == 2, "pipelineDispose must fire once per dispose() call; got \(disposeCount)")
  }
}
