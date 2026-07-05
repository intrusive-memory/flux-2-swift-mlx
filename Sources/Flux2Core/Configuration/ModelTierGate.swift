// ModelTierGate.swift - Typed, tier-aware model refusal (Sortie A2, R5.1/R5.2)
// Copyright 2025 intrusive-memory

import Foundation

extension Flux2Model {
  /// Whether this is one of the Klein 9B variants (`klein9B`, `klein9BBase`,
  /// `klein9BKV`).
  ///
  /// Klein 9B is excluded on **every** platform and **every** memory tier —
  /// including 16 GB — by product decision (EXECUTION_PLAN.md §0 standing
  /// constraint). Its refusal is intentionally **not** memory-gated.
  public var isKlein9BVariant: Bool {
    switch self {
    case .klein9B, .klein9BBase, .klein9BKV: return true
    case .dev, .klein4B, .klein4BBase: return false
    }
  }
}

/// Tier-aware gate that resolves a requested `Flux2Model` to the model that may
/// actually run on a given `MemoryConfig.MemoryTier`, or refuses it with the
/// typed `Flux2Error.modelNotSupportedOnTier`.
///
/// Refusal rules (in order):
/// 1. **Klein 9B (all variants) — refused unconditionally on every tier**,
///    including 16 GB. This is a product decision, not a memory ceiling.
/// 2. **Dev (32B) — refused on the iPad tier** (hardware grounds: it exceeds
///    iPad-class unified memory). Typed refusal, not an OOM.
/// 3. **iPad tier forces `.klein4B`** — the only supported image model on iPad.
/// 4. **Mac tier passes the (non-9B) requested model through unchanged.**
///
/// Every refusal emits `Flux2TelemetryEvent.errorThrown` immediately before the
/// `throw` (CLAUDE.md §5a chokepoint convention: exactly one `errorThrown` per
/// throw site). The emit targets the injected `telemetry` reporter if supplied,
/// otherwise the process-wide `Flux2Telemetry.current` seam.
public enum ModelTierGate {

  /// Resolve `requested` for `tier`, or throw `Flux2Error.modelNotSupportedOnTier`.
  ///
  /// - Parameters:
  ///   - requested: The model the caller asked for.
  ///   - tier: The active device memory tier.
  ///   - telemetry: Optional reporter override; defaults to the process-wide
  ///     `Flux2Telemetry.current` seam when `nil`.
  /// - Returns: The model that may actually run (`.klein4B` on the iPad tier;
  ///   the requested model on the Mac tier).
  /// - Throws: `Flux2Error.modelNotSupportedOnTier` for Klein 9B (any tier) or
  ///   Dev (iPad tier).
  @discardableResult
  public static func resolve(
    _ requested: Flux2Model,
    forTier tier: MemoryConfig.MemoryTier,
    telemetry: (any Flux2TelemetryReporter)? = nil
  ) async throws -> Flux2Model {
    let sink = telemetry ?? Flux2Telemetry.current

    // 1. Klein 9B — refused UNCONDITIONALLY on every tier (product decision,
    //    NOT memory-gated). This must fire before any tier branching.
    if requested.isKlein9BVariant {
      let reason =
        "Klein 9B is excluded on every platform and tier by product decision. Use Klein 4B."
      await sink?.capture(
        .errorThrown(
          phase: .invalidConfiguration,
          errorDescription:
            "\(requested.displayName) refused on the \(tier.rawValue) tier: \(reason)"
        ))
      throw Flux2Error.modelNotSupportedOnTier(
        model: requested.rawValue, tier: tier.rawValue, reason: reason)
    }

    // 2. iPad tier — refuse Dev (32B) with the typed error (not an OOM).
    if tier == .iPad {
      if requested == .dev {
        let reason =
          "Flux.2 Dev (32B) exceeds iPad-class unified memory. Use Klein 4B."
        await sink?.capture(
          .errorThrown(
            phase: .invalidConfiguration,
            errorDescription:
              "\(requested.displayName) refused on the \(tier.rawValue) tier: \(reason)"
          ))
        throw Flux2Error.modelNotSupportedOnTier(
          model: requested.rawValue, tier: tier.rawValue, reason: reason)
      }

      // 3. Force Klein 4B on the iPad tier — the only supported image model.
      return .klein4B
    }

    // 4. Mac tier — pass the (already non-9B) requested model through unchanged.
    return requested
  }

  /// Convenience overload that resolves the tier from a RAM amount first.
  @discardableResult
  public static func resolve(
    _ requested: Flux2Model,
    forRAMGB ramGB: Int,
    telemetry: (any Flux2TelemetryReporter)? = nil
  ) async throws -> Flux2Model {
    try await resolve(
      requested, forTier: MemoryConfig.tier(forRAMGB: ramGB), telemetry: telemetry)
  }
}
