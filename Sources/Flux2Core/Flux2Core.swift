// Flux2Core - Swift MLX Framework for Flux.2 Image Generation
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Main entry point for Flux.2 image generation
public enum Flux2Core {
  public static let version = "3.4.0"
}

/// Errors that can occur during Flux.2 operations
public enum Flux2Error: Error, LocalizedError {
  case modelNotLoaded(String)
  case invalidConfiguration(String)
  case insufficientMemory(required: Int, available: Int)
  case weightLoadingFailed(String)
  case imageProcessingFailed(String)
  case generationFailed(String)
  case generationCancelled
  /// A requested model is refused on the active memory tier.
  ///
  /// This is a typed refusal, **not** an out-of-memory failure. It fires for
  /// two product/hardware reasons:
  /// - Klein 9B (all variants) is excluded on *every* tier by product decision
  ///   (unconditional; never memory-gated).
  /// - Flux.2 Dev (32B) is refused on the iPad tier on hardware grounds.
  case modelNotSupportedOnTier(model: String, tier: String, reason: String)

  public var errorDescription: String? {
    switch self {
    case .modelNotLoaded(let component):
      return "Model component not loaded: \(component)"
    case .invalidConfiguration(let message):
      return "Invalid configuration: \(message)"
    case .insufficientMemory(let required, let available):
      return
        "Insufficient memory: required \(required / 1_000_000_000)GB, available \(available / 1_000_000_000)GB"
    case .weightLoadingFailed(let message):
      return "Failed to load weights: \(message)"
    case .imageProcessingFailed(let message):
      return "Image processing failed: \(message)"
    case .generationFailed(let message):
      return "Generation failed: \(message)"
    case .generationCancelled:
      return "Generation was cancelled"
    case .modelNotSupportedOnTier(let model, let tier, let reason):
      return "Model \(model) is not supported on the \(tier) tier: \(reason)"
    }
  }
}
