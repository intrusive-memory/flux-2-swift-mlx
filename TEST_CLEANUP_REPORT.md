# Test Cleanup Report — OPERATION SILICON STETHOSCOPE Iteration 02

## Summary

Mission-added tests were reviewed against 12 CI-unsafe patterns. **No tests deleted.** One test flagged for review due to timing-sensitive assertions that may flake on a loaded CI runner.

---

## Removed

| File:Test | Reason | Confidence |
|-----------|--------|-----------|
| _(none)_ | N/A | N/A |

---

## Flagged for Review

| File:Test | Concern | Recommended Action |
|-----------|---------|-------------------|
| **Flux2TelemetryNoopOverheadTests.swift:testOverhead** | Timing-sensitive bounds on tight margins. The +1% Noop overhead bound = ~19μs headroom against a ~1.9ms/step baseline. On a loaded macos-26 CI runner, scheduler jitter alone can exceed 19μs and cause flake. Local agent run showed negative overhead (telemetry cost below ContinuousClock noise floor), but local quiet ≠ CI loaded. | Three options: (1) Move test to `make test-gpu` (local-only target, like existing Metal tests); (2) Add CI skip gate: `if ProcessInfo.processInfo.environment["CI"] != nil { Skip("timing test skipped in CI") }`; (3) Loosen bounds to +5%/+15% for CI tolerance. Recommend option (1) if the test is Sortie 9 validation only, or (2) if it must run in CI but with loose bounds. Document the choice in iter-03 EXECUTION_PLAN. |

---

## Files Reviewed

**Test Helpers (kept):**
- `Tests/TestHelpers/MockTelemetryReporter.swift` — Public actor mock. Not a test itself; used by contract tests. No CI issues.

**Test Suites (all kept):**
- `Tests/Flux2CoreTests/Flux2TelemetryAnomalyTests.swift` — 7 tests, synthetic-event contracts. No CI issues.
- `Tests/Flux2CoreTests/Flux2TelemetryDenoiseStepTests.swift` — 1 test, synthetic-event contracts. No CI issues.
- `Tests/Flux2CoreTests/Flux2TelemetryKVCacheHitTests.swift` — 2 tests, synthetic-event contracts. No CI issues.
- `Tests/Flux2CoreTests/Flux2TelemetryVAEDenormalizationTests.swift` — 2 tests, synthetic-event contracts. No CI issues.
- `Tests/Flux2CoreTests/Flux2TelemetryWeightLoadHistogramTests.swift` — 4 tests, uses `MLXArray.zeros()` allocation. F9/F10 compliant. No CI issues.
- `Tests/Flux2CoreTests/Flux2TelemetryNoopOverheadTests.swift` — 1 test, timing-sensitive. **Flagged** (see above).

---

## Build Verification

**Status:** SKIPPED (no deletions made)

Since no tests were deleted, `make test` was not run. All flagged tests remain in place and will be part of the default test suite until iter-03 explicitly gates or moves them.

---

## Recommendations for iter-03

The NoopOverhead test is a Sortie 9 delivery artifact validating that telemetry adds < 1% overhead (Noop) and < 5% overhead (Mock). This is a real engineering requirement from the mission specification, not a hypothesis test. However, +1% margin on a ~1.9ms baseline translates to ~19μs headroom — on the edge of ContinuousClock granularity and CI scheduler jitter. **Decision point for iter-03:** either gate it as `make test-gpu` (local-only, like existing Metal tests), add a CI skip, or loosen bounds for CI runs. The test itself is sound; the issue is environment sensitivity, not test logic. Recommend documenting the choice in iter-03 EXECUTION_PLAN before the next release cycle.
