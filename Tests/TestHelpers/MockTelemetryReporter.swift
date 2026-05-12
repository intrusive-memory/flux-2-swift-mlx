// MockTelemetryReporter.swift — shared mock for Flux2CoreTests telemetry suites.
// Copyright 2025 Vincent Gourbin

import Flux2Core
import Foundation

/// Public test mock used by Flux2CoreTests (Sortie 7a/7b contract tests) and by
/// Sortie 9's overhead test. Sortie 8's lock-contention test deliberately uses
/// a local mock instead (F11 — its scope is intentionally self-contained).
///
/// Implemented as an `actor` because emissions in `Flux2Pipeline` originate
/// from multiple concurrent contexts (`Task { ... }` wrappers in sync inits and
/// in `setTimesteps`, plus the async denoise loop bodies). An actor gives us a
/// `Sendable` reporter with safe `_events` append/read without a separate lock.
public actor MockTelemetryReporter: Flux2TelemetryReporter {
    private var _events: [Flux2TelemetryEvent] = []

    public init() {}

    public func capture(_ event: Flux2TelemetryEvent) async {
        _events.append(event)
    }

    /// Snapshot of every event captured so far. Already inside the actor's
    /// isolation, so callers from `await` contexts read a consistent slice.
    public func events() -> [Flux2TelemetryEvent] {
        _events
    }

    public func reset() {
        _events.removeAll()
    }
}
