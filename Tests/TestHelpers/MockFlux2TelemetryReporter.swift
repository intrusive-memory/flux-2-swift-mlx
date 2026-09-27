import Flux2Core

/// Test-only reporter that captures every Flux2TelemetryEvent into an
/// append-only log. Not part of the public API.
///
/// The append-only log is implemented as an actor so concurrent capture
/// calls from the fire-and-forget Task in `Flux2Pipeline.init` and other
/// emission sites are serialized without data races.
///
/// ## Waiting for fire-and-forget dispatch
///
/// `pipelineInit` is dispatched via a single unstructured `Task { ... }` at
/// the end of `Flux2Pipeline.init`. After the sync init returns the Task may
/// not yet have delivered its event to this actor. Don't use a fixed sleep as
/// a barrier (it flakes on loaded CI runners); await the condition instead:
///
///     let sawInit = await reporter.waitFor { events in
///       events.contains { if case .pipelineInit = $0 { true } else { false } }
///     }
///     #expect(sawInit)
///
/// `waitFor` returns as soon as the predicate holds and only runs to its
/// (generous) timeout when the event never arrives, i.e. when the test fails.
public actor MockFlux2TelemetryReporter: Flux2TelemetryReporter {
  private(set) var events: [Flux2TelemetryEvent] = []

  public init() {}

  public func capture(_ event: Flux2TelemetryEvent) async {
    events.append(event)
  }

  public func snapshot() async -> [Flux2TelemetryEvent] {
    events
  }

  public func clear() async {
    events.removeAll()
  }

  /// Suspend until the captured log satisfies `predicate`, or `timeout`
  /// elapses.
  /// - Returns: `true` if the predicate was satisfied before the deadline.
  public func waitFor(
    timeout: Duration = .seconds(10),
    _ predicate: @Sendable ([Flux2TelemetryEvent]) -> Bool
  ) async -> Bool {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while !predicate(events) {
      if clock.now >= deadline { return false }
      try? await Task.sleep(for: .milliseconds(5))
    }
    return true
  }
}
