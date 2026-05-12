// MockTelemetryReporter.swift — Recording mock for Flux2TelemetryReporter.
// Used by Flux2CoreTests (Sortie 7a/7b) and Flux2CoreTests (Sortie 8).
// Thread-safe via Swift actor isolation.

import Flux2Core

/// A recording `Flux2TelemetryReporter` for use in unit tests.
///
/// Every `capture(_:)` call appends the event to an internal array.
/// Tests retrieve the recorded events via `events()`, or narrow them
/// by case via `events(matching:)`.
///
/// Being an `actor` means the internal array requires no additional locking;
/// all access is serialised through the actor's executor.
public actor MockTelemetryReporter: Flux2TelemetryReporter {

    // MARK: - Storage

    private var _events: [Flux2TelemetryEvent] = []

    // MARK: - init

    public init() {}

    // MARK: - Flux2TelemetryReporter

    /// Append `event` to the recorded event list.
    public func capture(_ event: Flux2TelemetryEvent) async {
        _events.append(event)
    }

    // MARK: - Accessors

    /// Return all recorded events in insertion order.
    public func events() -> [Flux2TelemetryEvent] {
        _events
    }

    /// Remove all recorded events. Useful between successive assertions in one test.
    public func reset() {
        _events.removeAll()
    }
}
