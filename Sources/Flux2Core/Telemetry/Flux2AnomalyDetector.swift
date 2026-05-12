import Foundation
import Tuberia

/// Heuristic anomaly detector for tensor stats. Lives in flux (not SwiftTuberia)
/// per EXECUTION_PLAN Q6 — avoids expanding the SwiftTuberia API for one consumer's heuristic.
public enum Flux2AnomalyDetector {
    /// Default threshold for the .outOfRange check. Matches TuberiaTensorStat.defaultOutOfRangeThreshold
    /// (|x| > 1e6) for fp numerical stability. Override per-call via `outOfRangeThreshold:` if needed.
    public static let defaultOutOfRangeThreshold: Double = TuberiaTensorStat.defaultOutOfRangeThreshold

    /// Returns the list of anomalies detected in `stat`.
    /// - Parameters:
    ///   - stat: the tensor stat to inspect
    ///   - checkZeroLatent: pass `true` only for latents/embeddings (where near-zero mean+std signals collapse)
    ///   - expectedDtype: if provided and the stat's dtype mismatches, emits `.dtypeUnexpected`
    public static func anomalies(
        in stat: TuberiaTensorStat,
        checkZeroLatent: Bool = false,
        expectedDtype: String? = nil
    ) -> [Flux2TelemetryEvent.AnomalyKind] {
        var kinds: [Flux2TelemetryEvent.AnomalyKind] = []
        if stat.hasNaN { kinds.append(.nan) }
        if stat.hasInf { kinds.append(.inf) }
        if abs(stat.max) > defaultOutOfRangeThreshold || abs(stat.min) > defaultOutOfRangeThreshold {
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
