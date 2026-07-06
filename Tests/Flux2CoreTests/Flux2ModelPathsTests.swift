// Flux2ModelPathsTests.swift - Regression tests for Flux2ModelPaths path resolution.

import Foundation
import Testing

@testable import Flux2Core

@Suite struct Flux2ModelPathsTests {

  /// Create a unique temp directory and clean it up automatically.
  private func makeTempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("flux2-paths-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  private func touch(_ url: URL) throws {
    try Data().write(to: url)
  }

  // Regression: pre-quantized MLX transformers (themindstudio/flux2-klein-4b-mlx-4bit,
  // `.klein4B_4bit`) ship a `transformer/` subfolder with a safetensors index and
  // sharded weights but NO config.json / model_index.json. The resolved-depth marker
  // check must accept them, otherwise loadTransformer() throws modelNotLoaded even
  // though the weights are fully present.
  @Test func preQuantizedMLXTransformerAcceptsSafetensorsIndex() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }

    try touch(dir.appendingPathComponent("model.safetensors.index.json"))
    try touch(dir.appendingPathComponent("0.safetensors"))

    #expect(
      Flux2ModelPaths.hasResolvedDepthMarker(at: dir, for: .transformer(.klein4B_4bit)) == true)
  }

  // A bare `*.safetensors` file (no index) is also accepted for pre-quantized MLX.
  @Test func preQuantizedMLXTransformerAcceptsBareSafetensors() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }

    try touch(dir.appendingPathComponent("0.safetensors"))

    #expect(
      Flux2ModelPaths.hasResolvedDepthMarker(at: dir, for: .transformer(.klein4B_4bit)) == true)
  }

  // An empty / incomplete directory still returns false.
  @Test func emptyDirectoryRejected() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }

    #expect(
      Flux2ModelPaths.hasResolvedDepthMarker(at: dir, for: .transformer(.klein4B_4bit)) == false)
  }

  // The relaxed rule must NOT leak to non-pre-quantized transformers: a bf16
  // transformer dir with only safetensors (no config.json) is still rejected.
  @Test func nonPreQuantizedTransformerStillRequiresConfig() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }

    try touch(dir.appendingPathComponent("model.safetensors.index.json"))
    try touch(dir.appendingPathComponent("0.safetensors"))

    #expect(
      Flux2ModelPaths.hasResolvedDepthMarker(at: dir, for: .transformer(.klein4B_bf16)) == false)
  }

  // The strict marker (config.json) is still accepted for any component.
  @Test func configJsonAcceptedForNonPreQuantized() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }

    try touch(dir.appendingPathComponent("config.json"))

    #expect(
      Flux2ModelPaths.hasResolvedDepthMarker(at: dir, for: .transformer(.klein4B_bf16)) == true)
  }
}
