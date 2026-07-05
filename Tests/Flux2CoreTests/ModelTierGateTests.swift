// ModelTierGateTests.swift
// Sortie A2: typed, tier-aware model refusal.
//
// Proves:
//  - All three Klein 9B variants are refused UNCONDITIONALLY (16 GB AND 8 GB
//    iPad tiers, and the Mac tier) — refusal is a product decision, not
//    memory-gated.
//  - Dev (32B) is refused on the iPad tier with the typed error (not an OOM).
//  - Klein 4B is accepted on every tier, and the iPad tier forces .klein4B.
//  - Every refusal emits .errorThrown immediately before the throw
//    (CLAUDE.md §5a: one errorThrown per throw site).

import Flux2Core
import TestHelpers
import Testing

@Suite("Model Tier Gate (Sortie A2)")
struct ModelTierGateTests {

  private static let klein9BVariants: [Flux2Model] = [.klein9B, .klein9BBase, .klein9BKV]

  // MARK: - Klein 9B refused unconditionally (NOT memory-gated)

  /// The 16 GB tier is the load-bearing case: it proves refusal is a product
  /// decision, not a memory ceiling. 16 GB is plenty of RAM — Klein 9B is still
  /// refused.
  @Test func klein9BVariantsRefusedOn16GBTier() async {
    for variant in Self.klein9BVariants {
      await #expect(throws: Flux2Error.self) {
        try await ModelTierGate.resolve(variant, forRAMGB: 16)
      }
    }
  }

  @Test func klein9BVariantsRefusedOn8GBTier() async {
    for variant in Self.klein9BVariants {
      await #expect(throws: Flux2Error.self) {
        try await ModelTierGate.resolve(variant, forRAMGB: 8)
      }
    }
  }

  /// Even the roomy Mac tier (64 GB) refuses Klein 9B — proving the refusal is
  /// unconditional across every tier, not tied to available memory.
  @Test func klein9BVariantsRefusedOnMacTier() async {
    for variant in Self.klein9BVariants {
      await #expect(throws: Flux2Error.self) {
        try await ModelTierGate.resolve(variant, forRAMGB: 64)
      }
    }
  }

  /// Assert the thrown error is specifically `.modelNotSupportedOnTier` (not an
  /// OOM / insufficientMemory), for all three variants on the 16 GB tier.
  @Test func klein9BThrowsTypedModelNotSupportedOn16GB() async throws {
    for variant in Self.klein9BVariants {
      var caught: Flux2Error?
      do {
        _ = try await ModelTierGate.resolve(variant, forRAMGB: 16)
      } catch let error as Flux2Error {
        caught = error
      }
      guard case .modelNotSupportedOnTier(let model, let tier, _) = caught else {
        Issue.record(
          "Expected .modelNotSupportedOnTier for \(variant), got: \(String(describing: caught))")
        continue
      }
      #expect(model == variant.rawValue)
      #expect(tier == MemoryConfig.MemoryTier.iPad.rawValue)
    }
  }

  // MARK: - Dev refused on the iPad tier (typed, not OOM)

  @Test func devRefusedOnIPadTier() async throws {
    var caught: Flux2Error?
    do {
      _ = try await ModelTierGate.resolve(.dev, forRAMGB: 16)
    } catch let error as Flux2Error {
      caught = error
    }
    guard case .modelNotSupportedOnTier(let model, let tier, _) = caught else {
      Issue.record("Expected .modelNotSupportedOnTier for Dev, got: \(String(describing: caught))")
      return
    }
    #expect(model == Flux2Model.dev.rawValue)
    #expect(tier == MemoryConfig.MemoryTier.iPad.rawValue)
  }

  /// Dev is a valid choice on the Mac tier — it must NOT be refused there.
  @Test func devAcceptedOnMacTier() async throws {
    let resolved = try await ModelTierGate.resolve(.dev, forRAMGB: 64)
    #expect(resolved == .dev)
  }

  // MARK: - Klein 4B accepted; iPad forces Klein 4B

  @Test func klein4BAcceptedOnIPadTier() async throws {
    let resolved = try await ModelTierGate.resolve(.klein4B, forRAMGB: 16)
    #expect(resolved == .klein4B)
  }

  @Test func klein4BAcceptedOn8GBTier() async throws {
    let resolved = try await ModelTierGate.resolve(.klein4B, forRAMGB: 8)
    #expect(resolved == .klein4B)
  }

  @Test func klein4BAcceptedOnMacTier() async throws {
    let resolved = try await ModelTierGate.resolve(.klein4B, forRAMGB: 64)
    #expect(resolved == .klein4B)
  }

  /// The iPad tier forces `.klein4B` even when a different (non-refused)
  /// inference-capable variant is requested — e.g. the Klein 4B base model.
  @Test func iPadTierForcesKlein4B() async throws {
    let resolved = try await ModelTierGate.resolve(.klein4BBase, forRAMGB: 16)
    #expect(resolved == .klein4B)
  }

  // MARK: - errorThrown precedes every throw (CLAUDE.md §5a)

  @Test func klein9BRefusalEmitsErrorThrown() async throws {
    for variant in Self.klein9BVariants {
      let reporter = MockFlux2TelemetryReporter()
      _ = try? await ModelTierGate.resolve(variant, forRAMGB: 16, telemetry: reporter)
      let events = await reporter.snapshot()
      let emitted = events.contains { event in
        if case .errorThrown(phase: .invalidConfiguration, _) = event { return true }
        return false
      }
      #expect(
        emitted,
        "Expected .errorThrown before refusing \(variant); got: \(events)")
    }
  }

  @Test func devRefusalEmitsErrorThrown() async throws {
    let reporter = MockFlux2TelemetryReporter()
    _ = try? await ModelTierGate.resolve(.dev, forRAMGB: 16, telemetry: reporter)
    let events = await reporter.snapshot()
    let emitted = events.contains { event in
      if case .errorThrown(phase: .invalidConfiguration, _) = event { return true }
      return false
    }
    #expect(emitted, "Expected .errorThrown before refusing Dev on the iPad tier; got: \(events)")
  }

  /// The accept path must NOT emit an errorThrown (no throw ⇒ no emit).
  @Test func acceptedModelEmitsNoErrorThrown() async throws {
    let reporter = MockFlux2TelemetryReporter()
    _ = try await ModelTierGate.resolve(.klein4B, forRAMGB: 16, telemetry: reporter)
    let events = await reporter.snapshot()
    let emitted = events.contains { event in
      if case .errorThrown = event { return true }
      return false
    }
    #expect(!emitted, "Accepting Klein 4B must not emit .errorThrown; got: \(events)")
  }
}
