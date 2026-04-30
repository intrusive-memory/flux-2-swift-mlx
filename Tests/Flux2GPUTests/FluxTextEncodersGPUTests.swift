import Foundation
import MLX
import TestHelpers
// FluxTextEncodersGPUTests.swift — GPU-gated FluxTextEncoders integration tests
// API notes:
//   - FluxTextEncoders.shared.loadKleinModel(variant:from:) loads the Qwen3 model
//   - FluxTextEncoders.shared.extractKleinEmbeddings(prompt:) returns MLXArray [1, 512, 7680] for Klein 4B
//   - FluxTextEncoders.shared.generateQwen3(prompt:parameters:) returns GenerationResult with .text
//   - MLX.isNaN(_:) / MLX.isInf(_:) return MLXArray; .all().item() returns Bool
import Testing

@testable import Flux2Core
@testable import FluxTextEncoders

@Suite("FluxTextEncoders GPU")
struct FluxTextEncodersGPUTests {

  // Minimum memory: 16 GiB for Qwen3-4B-8bit (model ~4 GB + working memory)
  private static let minimumBytes: UInt64 = 16 * 1_073_741_824

  // Shared model path environment variable; must be set for real GPU runs
  private static var kleinModelPath: String? {
    ProcessInfo.processInfo.environment["KLEIN_MODEL_PATH"]
  }

  // MARK: - Test 1: Qwen3 embedding shape

  @Test(.timeLimit(.minutes(3))) func qwen3EmbeddingShape() async throws {
    guard checkGPUPreconditions(minimumBytes: Self.minimumBytes) else { return }
    guard let modelPath = Self.kleinModelPath else {
      // No model path on CI — precondition guard exits gracefully
      return
    }

    let encoders = FluxTextEncoders.shared
    try await encoders.loadKleinModel(variant: .klein4B, from: modelPath)
    defer { Task { @MainActor in encoders.unloadKleinModel() } }

    let embeddings = try encoders.extractKleinEmbeddings(
      prompt: "a red balloon floating in the sky")
    // Klein 4B: [1, 512, 7680]
    #expect(embeddings.shape.count == 3, "Expected 3D embedding tensor")
    #expect(embeddings.shape[0] == 1, "Batch dimension should be 1")
    #expect(embeddings.shape[1] == 512, "Sequence length should be 512")
    #expect(
      embeddings.shape[2] == KleinVariant.klein4B.outputDimension,
      "Hidden dim should match Klein 4B output dimension")
  }

  // MARK: - Test 2: Embedding values are finite

  @Test(.timeLimit(.minutes(3))) func embeddingValuesAreFinite() async throws {
    guard checkGPUPreconditions(minimumBytes: Self.minimumBytes) else { return }
    guard let modelPath = Self.kleinModelPath else {
      return
    }

    let encoders = FluxTextEncoders.shared
    try await encoders.loadKleinModel(variant: .klein4B, from: modelPath)
    defer { Task { @MainActor in encoders.unloadKleinModel() } }

    let embeddings = try encoders.extractKleinEmbeddings(prompt: "a cat sitting on a wooden table")

    let hasNaN = isNaN(embeddings).any().item(Bool.self)
    let hasInf = isInf(embeddings).any().item(Bool.self)

    #expect(!hasNaN, "Embeddings must not contain NaN values")
    #expect(!hasInf, "Embeddings must not contain Inf values")
  }

  // MARK: - Test 3: Generate returns non-empty text

  @Test(.timeLimit(.minutes(3))) func generateReturnsNonEmptyText() async throws {
    guard checkGPUPreconditions(minimumBytes: Self.minimumBytes) else { return }
    guard let modelPath = Self.kleinModelPath else {
      return
    }

    let encoders = FluxTextEncoders.shared
    try await encoders.loadKleinModel(variant: .klein4B, from: modelPath)
    defer { Task { @MainActor in encoders.unloadKleinModel() } }

    let result = try encoders.generateQwen3(
      prompt: "What is 2 + 2?",
      parameters: .greedy
    )

    #expect(!result.text.isEmpty, "generateQwen3 must return non-empty text")
    #expect(result.generatedTokens > 0, "At least one token should be generated")
  }

  // MARK: - Test 4: Klein embeddings are deterministic

  @Test(.timeLimit(.minutes(3))) func kleinEmbeddingsAreDeterministic() async throws {
    guard checkGPUPreconditions(minimumBytes: Self.minimumBytes) else { return }
    guard let modelPath = Self.kleinModelPath else {
      return
    }

    let encoders = FluxTextEncoders.shared
    try await encoders.loadKleinModel(variant: .klein4B, from: modelPath)
    defer { Task { @MainActor in encoders.unloadKleinModel() } }

    let prompt = "a mountain landscape at sunset with snow-capped peaks"
    let embeddings1 = try encoders.extractKleinEmbeddings(prompt: prompt)
    let embeddings2 = try encoders.extractKleinEmbeddings(prompt: prompt)

    // Both calls should return identical shapes
    #expect(
      embeddings1.shape == embeddings2.shape,
      "Repeated extractKleinEmbeddings calls should return same shape")

    // Values should be bitwise-equal (deterministic on same hardware)
    let equal = (embeddings1 .== embeddings2).all().item(Bool.self)
    #expect(equal, "Klein embeddings should be deterministic for identical inputs")
  }
}
