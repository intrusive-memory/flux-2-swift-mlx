public protocol Flux2TelemetryReporter: Sendable {
    func capture(_ event: Flux2TelemetryEvent) async
}

public struct NoopFlux2TelemetryReporter: Flux2TelemetryReporter {
    public init() {}
    public func capture(_ event: Flux2TelemetryEvent) async {}
}
