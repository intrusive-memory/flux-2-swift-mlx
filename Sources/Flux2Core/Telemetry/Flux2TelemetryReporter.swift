import Foundation

/// Sink for `Flux2TelemetryEvent` emissions. Implemented by the host adapter
/// (e.g. Vinetas) to fan flux events into the shared telemetry pipeline.
///
/// The reporter is installed on `Flux2Pipeline` (and propagated by it to all
/// owned subcomponents) via `setTelemetry(_:)`. See
/// `REQUIREMENTS-instrumentation.md` §3.2 and §4.
public protocol Flux2TelemetryReporter: Sendable {
    func capture(_ event: Flux2TelemetryEvent) async
}

/// No-op reporter used by tests and by hosts that want the guard branch to
/// pass (so emissions are wired) but do not want events to go anywhere.
///
/// Sortie 9's overhead test uses this to validate the +1% Noop-overhead bound.
///
/// Defined as a `struct` per `REQUIREMENTS-instrumentation.md` §3.2.
public struct NoopFlux2TelemetryReporter: Flux2TelemetryReporter {
    public init() {}
    public func capture(_ event: Flux2TelemetryEvent) async {}
}
