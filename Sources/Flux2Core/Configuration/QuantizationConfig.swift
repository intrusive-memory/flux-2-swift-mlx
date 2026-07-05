// QuantizationConfig.swift - Fine-grained quantization configuration
// Copyright 2025 Vincent Gourbin

import Foundation

/// Quantization level for the Mistral text encoder
public enum MistralQuantization: String, CaseIterable, Codable, Sendable {
  case bf16 = "bf16"  // Full precision ~48GB
  case mlx8bit = "8bit"  // 8-bit ~25GB
  case mlx6bit = "6bit"  // 6-bit ~19GB
  case mlx4bit = "4bit"  // 4-bit ~14GB

  public var estimatedMemoryGB: Int {
    switch self {
    case .bf16: return 48
    case .mlx8bit: return 25
    case .mlx6bit: return 19
    case .mlx4bit: return 14
    }
  }

  public var displayName: String {
    switch self {
    case .bf16: return "Full Precision (bf16)"
    case .mlx8bit: return "8-bit"
    case .mlx6bit: return "6-bit"
    case .mlx4bit: return "4-bit"
    }
  }
}

/// Quantization level for the Flux.2 diffusion transformer
public enum TransformerQuantization: String, CaseIterable, Codable, Sendable {
  case bf16 = "bf16"  // Full precision ~64GB
  case qint8 = "qint8"  // 8-bit ~32GB
  case int4 = "int4"  // 4-bit ~16GB (on-the-fly quantization)

  public var estimatedMemoryGB: Int {
    switch self {
    case .bf16: return 64
    case .qint8: return 32
    case .int4: return 16
    }
  }

  public var displayName: String {
    switch self {
    case .bf16: return "Full Precision (bf16)"
    case .qint8: return "8-bit (qint8)"
    case .int4: return "4-bit (int4)"
    }
  }

  public var bits: Int {
    switch self {
    case .bf16: return 16
    case .qint8: return 8
    case .int4: return 4
    }
  }

  public var groupSize: Int { 64 }
}

/// Independent quantization configuration for text encoder and transformer
public struct Flux2QuantizationConfig: Codable, Sendable {
  /// Quantization for the Mistral text encoder
  public var textEncoder: MistralQuantization

  /// Quantization for the Flux.2 diffusion transformer
  public var transformer: TransformerQuantization

  public init(
    textEncoder: MistralQuantization,
    transformer: TransformerQuantization
  ) {
    self.textEncoder = textEncoder
    self.transformer = transformer
  }

  // MARK: - Working-set overhead (Sortie B3, OQ-4)

  /// Working-set overhead (GB) added on top of the larger of the text-encoder /
  /// transformer weight budgets to estimate peak resident memory — **Mac tier**.
  ///
  /// This replaces the historical magic `+8`, whose inline comment read
  /// "VAE 3 + working 5". Two named inputs revise that derivation:
  ///
  ///   • **VAE (A7 correction).** The Klein 4B `vae/` subfolder on the CDN is
  ///     ~0.168 GB, NOT the ~3 GB the old `+8` assumed — a ~2.8 GB
  ///     overstatement (see `CDN_PROVISIONING.md`, "Discrepancy note").
  ///   • **Working / scratch buffers** (denoise activations + VAE-decode
  ///     activations): unmeasured on-device as of B3.
  ///
  /// Per OQ-4 we ship a NAMED, DOCUMENTED constant now and keep a CONSERVATIVE
  /// floor; the precise value is set from measured on-device A6/A8
  /// `phys_footprint` telemetry (16 GB hardware, extrapolated via the §2
  /// max-phase model) in a FOLLOW-UP once the measurement lands. No on-device
  /// number exists yet — the A8 smoke test skips headless when weights are
  /// absent. We therefore deliberately do **not** bank the ~2.8 GB the VAE
  /// correction frees on the Mac tier: this floor stays at the historical
  /// **8 GB** (VAE-decode headroom + ~5 GB working, generously rounded), so the
  /// Mac estimate is unchanged (no regression).
  public static let macWorkingSetOverheadGB = 8

  /// Working-set overhead (GB) — **iPad tier** (`.iPad` and the `.iPad8GB`
  /// sub-tier).
  ///
  /// iPad generation is driven at low resolution (≤1024², and 512² on the 8 GB
  /// sub-tier once B4 lands), so the denoise/VAE activation buffers are
  /// materially smaller than the Mac working budget. Applying the A7 VAE
  /// correction we budget ~1 GB for the VAE (0.168 GB of weights rounded up to
  /// cover decode activations) + ~5 GB working = **6 GB**. This is still a
  /// conservative floor — it does NOT drop to an aggressive unmeasured value;
  /// it only removes the ~2 GB the corrected VAE figure no longer needs. Like
  /// the Mac constant, the precise value is pending the A6/A8 measurement.
  public static let iPadWorkingSetOverheadGB = 6

  /// Total estimated memory requirement in GB.
  ///
  /// Text encoder and transformer are never resident simultaneously, so we take
  /// the max of the two weight budgets and add the working-set overhead. This
  /// property uses the conservative **Mac** overhead so existing callers and
  /// estimates are unchanged; tier-aware callers should use
  /// `estimatedTotalMemoryGB(forTier:)`.
  public var estimatedTotalMemoryGB: Int {
    max(textEncoder.estimatedMemoryGB, transformer.estimatedMemoryGB)
      + Self.macWorkingSetOverheadGB
  }

  /// Tier-aware total estimated memory requirement in GB.
  ///
  /// Uses the named, documented working-set overhead for the given tier
  /// (`macWorkingSetOverheadGB` / `iPadWorkingSetOverheadGB`) rather than the
  /// retired Mac-shaped `+8` guess.
  public func estimatedTotalMemoryGB(forTier tier: MemoryConfig.MemoryTier) -> Int {
    let overhead: Int
    switch tier {
    case .iPad, .iPad8GB: overhead = Self.iPadWorkingSetOverheadGB
    case .mac: overhead = Self.macWorkingSetOverheadGB
    }
    return max(textEncoder.estimatedMemoryGB, transformer.estimatedMemoryGB) + overhead
  }

  /// Peak memory during text encoding phase
  public var textEncodingPhaseMemoryGB: Int {
    textEncoder.estimatedMemoryGB + 1  // +1GB for embeddings
  }

  /// Peak memory during image generation phase
  public var imageGenerationPhaseMemoryGB: Int {
    transformer.estimatedMemoryGB + 3 + 5  // +3GB VAE, +5GB working memory
  }

  // MARK: - Presets

  /// High quality preset - requires ~90GB+ RAM
  public static let highQuality = Flux2QuantizationConfig(
    textEncoder: .bf16,
    transformer: .bf16
  )

  /// Balanced preset - requires ~57GB RAM (recommended for 64GB Macs)
  public static let balanced = Flux2QuantizationConfig(
    textEncoder: .mlx8bit,
    transformer: .qint8
  )

  /// Memory efficient preset - requires ~47GB RAM
  public static let memoryEfficient = Flux2QuantizationConfig(
    textEncoder: .mlx4bit,
    transformer: .qint8
  )

  /// Minimal preset - requires ~47GB RAM
  public static let minimal = Flux2QuantizationConfig(
    textEncoder: .mlx4bit,
    transformer: .qint8
  )

  /// Ultra-minimal preset - requires ~30GB RAM (4-bit transformer)
  public static let ultraMinimal = Flux2QuantizationConfig(
    textEncoder: .mlx4bit,
    transformer: .int4
  )

  /// Default preset (balanced)
  public static let `default` = balanced
}

extension Flux2QuantizationConfig: CustomStringConvertible {
  public var description: String {
    "Flux2QuantizationConfig(text: \(textEncoder.rawValue), transformer: \(transformer.rawValue), ~\(estimatedTotalMemoryGB)GB)"
  }
}
