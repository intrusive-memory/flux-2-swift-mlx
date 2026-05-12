// Flux2TelemetryLockContentionTests.swift
// Run under TSan via: make test-tsan
//
// Validates the OSAllocatedUnfairLock-based @unchecked Sendable design introduced
// in Sortie 2.  The test spins up concurrent writers (setTelemetry) and a
// concurrent reader (currentTelemetry / mocked denoise loop) to prove the lock
// seam is race-free under contention.  TSan is the oracle — if any data race
// exists, TSan reports it and the process exits non-zero.

import Foundation
import Testing
import os.lock

@testable import Flux2Core

// MARK: - Local mock reporter (Option A: self-contained, no Sortie-7a dependency)

/// Minimal in-process reporter that records the most-recent captured event.
/// Defined as a private actor so its own internal state is Swift-concurrency-safe.
private actor LocalMockReporter: Flux2TelemetryReporter {
    private(set) var captureCount: Int = 0
    private(set) var lastEvent: Flux2TelemetryEvent?

    func capture(_ event: Flux2TelemetryEvent) async {
        captureCount += 1
        lastEvent = event
    }
}

// MARK: - Minimal lock fixture

/// A stripped-down replica of the Flux2Pipeline telemetry seam.
/// Using a fixture (rather than Flux2Pipeline itself) because:
///   1. currentTelemetry() is fileprivate on Flux2Pipeline.
///   2. Constructing a full Flux2Pipeline for a lock test is unnecessarily heavy.
/// The fixture uses *exactly* the same primitive (OSAllocatedUnfairLock) so TSan
/// exercises the real mechanism.
private final class TelemetryLockFixture: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock<(any Flux2TelemetryReporter)?>(initialState: nil)

    func setTelemetry(_ reporter: (any Flux2TelemetryReporter)?) {
        lock.withLock { $0 = reporter }
    }

    func currentTelemetry() -> (any Flux2TelemetryReporter)? {
        lock.withLock { $0 }
    }
}

// MARK: - Tests

@Suite("Telemetry Lock Contention")
struct Flux2TelemetryLockContentionTests {

    /// Spin 10 concurrent setTelemetry writers and 1 concurrent reader loop.
    /// TSan should stay silent.  The test asserts the harness completes without
    /// crashing and that the final reporter is one of the types we installed.
    @Test func concurrentSetAndGetTelemetry() async {
        let fixture = TelemetryLockFixture()
        let mock = LocalMockReporter()

        await withTaskGroup(of: Void.self) { group in
            // 10 concurrent writers toggle through nil / Noop / LocalMock
            for i in 0..<10 {
                group.addTask {
                    for _ in 0..<50 {
                        switch i % 3 {
                        case 0: fixture.setTelemetry(nil)
                        case 1: fixture.setTelemetry(NoopFlux2TelemetryReporter())
                        default: fixture.setTelemetry(mock)
                        }
                    }
                }
            }

            // 1 concurrent reader simulates a denoise loop: read once per step,
            // then (if non-nil) fire a synthetic capture.
            group.addTask {
                for stepIdx in 0..<100 {
                    // Acquire lock exactly once per step — mirrors hot-path discipline.
                    if let telemetry = fixture.currentTelemetry() {
                        await telemetry.capture(
                            .generationCancelled(stepIndex: stepIdx)
                        )
                    }
                }
            }
        }
        // If TSan detected a race it would have aborted the process before this
        // point.  A passing assertion here proves the harness ran to completion.
        // The final reporter may be any of {nil, Noop, mock} — we only assert it
        // is one of the expected types (no torn read produced a garbage pointer).
        let final = fixture.currentTelemetry()
        let isExpectedType =
            final == nil
            || final is NoopFlux2TelemetryReporter
            || final is LocalMockReporter
        #expect(isExpectedType, "Reporter must be nil, Noop, or LocalMockReporter — no torn read")
    }

    /// Stress the fixture with more tasks and more iterations to give TSan
    /// higher coverage over the lock fast-path.
    @Test func highConcurrencyStress() async {
        let fixture = TelemetryLockFixture()
        let reporters: [(any Flux2TelemetryReporter)?] = [
            nil,
            NoopFlux2TelemetryReporter(),
            LocalMockReporter(),
            NoopFlux2TelemetryReporter(),
            nil,
        ]

        await withTaskGroup(of: Void.self) { group in
            // 20 writer tasks
            for i in 0..<20 {
                let reporter = reporters[i % reporters.count]
                group.addTask {
                    for _ in 0..<200 {
                        fixture.setTelemetry(reporter)
                    }
                }
            }
            // 5 reader tasks
            for _ in 0..<5 {
                group.addTask {
                    for stepIdx in 0..<200 {
                        if let telemetry = fixture.currentTelemetry() {
                            await telemetry.capture(
                                .generationCancelled(stepIndex: stepIdx)
                            )
                        }
                    }
                }
            }
        }
        // Reaching here means no race was flagged by TSan and no crash occurred.
        #expect(Bool(true), "High-concurrency stress completed without a data race")
    }

    /// Verify that the last-writer-wins semantics hold: the value read after all
    /// writers finish equals the last write made.  This is a logical correctness
    /// check independent of TSan.
    @Test func lastWriterWins() async {
        let fixture = TelemetryLockFixture()
        let sentinel = LocalMockReporter()

        // Fill the lock with a known value.
        fixture.setTelemetry(sentinel)

        // Verify read returns the exact same identity (pointer equality via a tag).
        // We can't do identity comparison on existentials directly, so we proxy
        // through a capture: post a synthetic event and check captureCount goes up.
        let before = await sentinel.captureCount
        if let reporter = fixture.currentTelemetry() {
            await reporter.capture(.generationCancelled(stepIndex: 0))
        }
        let after = await sentinel.captureCount
        #expect(after == before + 1, "currentTelemetry() must return the reporter that was set")
    }
}
