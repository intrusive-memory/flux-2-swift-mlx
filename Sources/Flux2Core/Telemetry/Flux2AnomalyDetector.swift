// Flux2AnomalyDetector.swift
// Internal anomaly classifier used at every stat-carrying telemetry emission
// site to produce side-channel `numericalAnomaly` events.
//
// Q6: the helper lives in flux-2-swift-mlx, NOT in SwiftTuberia. Hosts
// (Vinetas) read events through `Flux2TelemetryReporter`; this helper is
// fired internally next to every stat-carrying emission.
//
// F4: The out-of-range threshold is referenced by NAME
// (`TuberiaTensorStat.defaultOutOfRangeThreshold`) — never by literal — so
// SwiftTuberia owns the constant and any future tuning lands there.

import Foundation
import Tuberia

/// Pure helper that classifies a `TuberiaTensorStat` snapshot into zero or
/// more `Flux2TelemetryEvent.AnomalyKind` cases. Emission sites loop over
/// the returned kinds and emit one `numericalAnomaly` event per kind.
public enum Flux2AnomalyDetector {

  /// Classify anomalies present in `stat`.
  ///
  /// - Parameters:
  ///   - stat: the sampled tensor statistic (already taken inside an
  ///     `if let telemetry = currentTelemetry()` guard at the call site).
  ///   - checkZeroLatent: set to `true` for latent tensors only (not for
  ///     embeddings or final RGB pixels). Defaults to `false`.
  ///   - expectedDtype: optional dtype string to compare against `stat.dtype`.
  ///     When non-nil and the strings differ, `.dtypeUnexpected` is reported.
  /// - Returns: array of distinct `AnomalyKind` cases observed; empty when
  ///   the tensor is healthy.
  public static func anomalies(
    in stat: TuberiaTensorStat,
    checkZeroLatent: Bool = false,
    expectedDtype: String? = nil
  ) -> [Flux2TelemetryEvent.AnomalyKind] {
    var kinds: [Flux2TelemetryEvent.AnomalyKind] = []
    if stat.hasNaN { kinds.append(.nan) }
    if stat.hasInf { kinds.append(.inf) }
    // F4: reference the threshold by NAME, never by literal.
    let threshold = TuberiaTensorStat.defaultOutOfRangeThreshold
    if abs(stat.max) > threshold || abs(stat.min) > threshold {
      kinds.append(.outOfRange)
    }
    if checkZeroLatent && abs(stat.mean) < 1e-6 && stat.std < 1e-6 {
      kinds.append(.zeroLatent)
    }
    if let expected = expectedDtype, stat.dtype != expected {
      kinds.append(.dtypeUnexpected)
    }
    return kinds
  }
}
