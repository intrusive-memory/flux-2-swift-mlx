/**
 * TextEncoderModelDownloader.swift
 * Downloads Mistral and Qwen3 models from the intrusive-memory CDN via SwiftAcervo.
 *
 * Sortie 18 (R2.5): Replaced legacy Hub-based downloads with SwiftAcervo's
 * manifest-driven CDN downloads. Storage location is `Acervo.sharedModelsDirectory`;
 * resolved via `ACERVO_APP_GROUP_ID` env var or entitlement (v0.10.0 — no fallback).
 * No override hook is exposed.
 *
 * Manifest-first file selection: passes `files: []` to Acervo so the per-repo
 * CDN manifest is the sole authoritative source for what gets downloaded.
 * Hardcoded request lists were retired because upstream repos diverge on
 * optional files (e.g., 4-bit Mistral has no `chat_template.jinja`; Qwen3
 * MLX repos ship neither `tokenizer.model` nor `generation_config.json`),
 * which made the legacy lists throw `AcervoError.fileNotInManifest`.
 */

import Foundation
import SwiftAcervo

/// Progress callback for download updates
public typealias TextEncoderDownloadProgressCallback = @Sendable (Double, String) -> Void

/// Model downloader backed by SwiftAcervo's CDN manifest pipeline.
public class TextEncoderModelDownloader {

  /// Optional auth token preserved on the type for API stability.
  /// SwiftAcervo downloads are unauthenticated CDN reads; this is retained
  /// so `init(hfToken:)` does not break consumers.
  private var hfToken: String?  // swift-format-ignore: name spelling preserved for API stability

  /// Canonical storage directory for downloaded models.
  ///
  /// Backed by `Acervo.sharedModelsDirectory`. As of SwiftAcervo v0.10.0
  /// the directory is resolved via `ACERVO_APP_GROUP_ID` env var or the
  /// `com.apple.security.application-groups` entitlement — no silent fallback.
  /// Sortie 14 removed the consumer override hook (`customModelsDirectory`);
  /// no replacement is exposed.
  public static var modelsDirectory: URL {
    Acervo.sharedModelsDirectory
  }

  public init(hfToken: String? = nil) {
    self.hfToken = hfToken
    if let token = hfToken {
      setenv("HF_TOKEN", token, 1)
    }
  }

  // MARK: - Mistral

  /// Check if a Mistral model is already downloaded.
  public static func isModelDownloaded(_ model: ModelInfo) -> Bool {
    return findModelPath(for: model) != nil
  }

  /// Find a Mistral model path in the Acervo shared models directory.
  /// Returns the directory only when `config.json` is present and the
  /// safetensors shard set is complete.
  public static func findModelPath(for model: ModelInfo) -> URL? {
    guard Acervo.isModelAvailable(model.repoId) else {
      return nil
    }
    guard let dir = try? Acervo.modelDirectory(for: model.repoId) else {
      return nil
    }
    let verification = verifyShardedModel(at: dir)
    return verification.complete ? dir : nil
  }

  /// Verify that a sharded model has all required safetensors files.
  ///
  /// Acervo's manifest verifies per-file SHA-256 + size. This check is
  /// retained as a complementary local-side shard-series sanity check
  /// (e.g., to catch a partially-deleted directory): index.json is not
  /// trusted; the shard filenames themselves declare totals.
  public static func verifyShardedModel(at path: URL) -> (complete: Bool, missing: [String]) {
    let contents = (try? FileManager.default.contentsOfDirectory(atPath: path.path)) ?? []
    let safetensorsFiles = contents.filter { $0.hasSuffix(".safetensors") }

    // Single file model
    if safetensorsFiles.contains("model.safetensors") {
      return (true, [])
    }

    // No safetensors files at all
    guard !safetensorsFiles.isEmpty else {
      return (false, ["No safetensors files found"])
    }

    // Parse sharded file pattern: model-XXXXX-of-YYYYY.safetensors
    var totalShards: Int?
    var foundIndices: Set<Int> = []

    for file in safetensorsFiles {
      let name = file.replacingOccurrences(of: ".safetensors", with: "")
      let parts = name.split(separator: "-")
      guard parts.count == 4,
        parts[0] == "model",
        parts[2] == "of",
        let index = Int(parts[1]),
        let total = Int(parts[3])
      else {
        continue
      }

      if totalShards == nil {
        totalShards = total
      } else if totalShards != total {
        return (false, ["Inconsistent shard totals: \(totalShards!) vs \(total)"])
      }

      foundIndices.insert(index)
    }

    if let total = totalShards {
      let expectedIndices = Set(1...total)
      let missing = expectedIndices.subtracting(foundIndices)

      if missing.isEmpty {
        return (true, [])
      } else {
        let missingFiles = missing.sorted().map {
          "model-\(String(format: "%05d", $0))-of-\(String(format: "%05d", total)).safetensors"
        }
        return (false, missingFiles)
      }
    }

    // Has some safetensors files but not in standard sharded format
    return (true, [])
  }

  /// Download a Mistral model via Acervo CDN.
  public func download(
    _ model: ModelInfo,
    progress: TextEncoderDownloadProgressCallback? = nil
  ) async throws -> URL {
    if let existingPath = Self.findModelPath(for: model) {
      progress?(1.0, "Model already downloaded")
      return existingPath
    }

    progress?(0.0, "Starting download of \(model.name)...")
    print("\nDownloading \(model.name) via Acervo CDN...")
    print("Repository: \(model.repoId)")

    try await Acervo.ensureAvailable(
      model.repoId,
      files: [],
      progress: { acervoProgress in
        let message =
          "Downloading \(acervoProgress.fileName) "
          + "(\(acervoProgress.fileIndex + 1)/\(acervoProgress.totalFiles))"
        progress?(acervoProgress.overallProgress, message)
      }
    )

    let modelUrl = try Acervo.modelDirectory(for: model.repoId)

    let verification = Self.verifyShardedModel(at: modelUrl)
    if !verification.complete {
      print("\nWarning: Local shard verification reports missing files: \(verification.missing)")
    }

    progress?(1.0, "Download complete!")
    print("\nDownload complete: \(modelUrl.path)")

    return modelUrl
  }

  /// Download a model by variant. Cut variants throw `notProvisionedOnCDN`.
  public func download(
    variant: ModelVariant,
    progress: TextEncoderDownloadProgressCallback? = nil
  ) async throws -> URL {
    if variant == .bf16 {
      throw TextEncoderModelDownloaderError.notProvisionedOnCDN(variant: .bf16)
    }
    guard let model = await TextEncoderModelRegistry.shared.model(withVariant: variant) else {
      throw TextEncoderModelDownloaderError.modelNotFound
    }
    return try await download(model, progress: progress)
  }

  // MARK: - Qwen3

  /// Download a Qwen3 model via Acervo CDN.
  public func downloadQwen3(
    _ model: Qwen3ModelInfo,
    progress: TextEncoderDownloadProgressCallback? = nil
  ) async throws -> URL {
    if let existingPath = Self.findQwen3ModelPath(for: model) {
      progress?(1.0, "Qwen3 model already downloaded")
      return existingPath
    }

    progress?(0.0, "Starting download of \(model.name)...")
    print("\nDownloading \(model.name) via Acervo CDN...")
    print("Repository: \(model.repoId)")

    try await Acervo.ensureAvailable(
      model.repoId,
      files: [],
      progress: { acervoProgress in
        let message =
          "Downloading \(acervoProgress.fileName) "
          + "(\(acervoProgress.fileIndex + 1)/\(acervoProgress.totalFiles))"
        progress?(acervoProgress.overallProgress, message)
      }
    )

    let modelUrl = try Acervo.modelDirectory(for: model.repoId)

    let verification = Self.verifyShardedModel(at: modelUrl)
    if !verification.complete {
      print(
        "\nWarning: Qwen3 local shard verification reports missing files: \(verification.missing)")
    }

    progress?(1.0, "Download complete!")
    print("\nQwen3 download complete: \(modelUrl.path)")

    return modelUrl
  }

  /// Download a Qwen3 model by variant.
  public func downloadQwen3(
    variant: Qwen3Variant,
    progress: TextEncoderDownloadProgressCallback? = nil
  ) async throws -> URL {
    guard let model = await TextEncoderModelRegistry.shared.qwen3Model(withVariant: variant) else {
      throw TextEncoderModelDownloaderError.qwen3ModelNotFound
    }
    return try await downloadQwen3(model, progress: progress)
  }

  /// Find a Qwen3 model path in the Acervo shared models directory.
  /// Returns the directory only when `config.json` is present and the
  /// safetensors shard set is complete.
  public static func findQwen3ModelPath(for model: Qwen3ModelInfo) -> URL? {
    guard Acervo.isModelAvailable(model.repoId) else {
      return nil
    }
    guard let dir = try? Acervo.modelDirectory(for: model.repoId) else {
      return nil
    }
    let verification = verifyShardedModel(at: dir)
    return verification.complete ? dir : nil
  }

  /// Check if a Qwen3 model is already downloaded.
  public static func isQwen3ModelDownloaded(_ model: Qwen3ModelInfo) -> Bool {
    return findQwen3ModelPath(for: model) != nil
  }

  /// Find a Qwen3 model path by variant.
  public static func findQwen3ModelPath(for variant: Qwen3Variant) -> URL? {
    let repoId = variant.repoId
    guard Acervo.isModelAvailable(repoId) else {
      return nil
    }
    guard let dir = try? Acervo.modelDirectory(for: repoId) else {
      return nil
    }
    let verification = verifyShardedModel(at: dir)
    return verification.complete ? dir : nil
  }

  /// Check if a Qwen3 model is already downloaded by variant.
  public static func isQwen3ModelDownloaded(variant: Qwen3Variant) -> Bool {
    return findQwen3ModelPath(for: variant) != nil
  }

  // MARK: - Direct repo / resolution

  /// Download a model by repo ID directly. Defers to the CDN manifest for the
  /// authoritative file list (`files: []`); the manifest is the sole source
  /// of truth for what a repo ships.
  public func downloadByRepoId(
    _ repoId: String,
    progress: TextEncoderDownloadProgressCallback? = nil
  ) async throws -> URL {
    progress?(0.0, "Starting download...")
    print("\nDownloading via Acervo CDN: \(repoId)")

    try await Acervo.ensureAvailable(
      repoId,
      files: [],
      progress: { acervoProgress in
        let message =
          "Downloading \(acervoProgress.fileName) "
          + "(\(acervoProgress.fileIndex + 1)/\(acervoProgress.totalFiles))"
        progress?(acervoProgress.overallProgress, message)
      }
    )

    let modelUrl = try Acervo.modelDirectory(for: repoId)
    progress?(1.0, "Download complete!")
    print("Model available at: \(modelUrl.path)")

    return modelUrl
  }

  /// Resolve a model identifier to a local path, downloading if necessary.
  public func resolveModel(
    _ identifier: String,
    progress: TextEncoderDownloadProgressCallback? = nil
  ) async throws -> URL {
    // Try to find by ID
    if let model = await TextEncoderModelRegistry.shared.model(withId: identifier) {
      if let existingPath = Self.findModelPath(for: model) {
        return existingPath
      }
      return try await download(model, progress: progress)
    }

    // Try variant matching
    if let variant = ModelVariant(rawValue: identifier) {
      if variant == .bf16 {
        throw TextEncoderModelDownloaderError.notProvisionedOnCDN(variant: .bf16)
      }
      if let model = await TextEncoderModelRegistry.shared.model(withVariant: variant) {
        if let existingPath = Self.findModelPath(for: model) {
          return existingPath
        }
        return try await download(model, progress: progress)
      }
    }

    // Check if it's a local path
    let localURL = URL(fileURLWithPath: identifier)
    if FileManager.default.fileExists(atPath: localURL.appendingPathComponent("config.json").path) {
      return localURL
    }

    // Try as a direct repo ID
    return try await downloadByRepoId(identifier, progress: progress)
  }

  /// Format bytes as human-readable string
  public static func formatSize(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.allowedUnits = [.useGB, .useMB]
    formatter.countStyle = .file
    return formatter.string(fromByteCount: bytes)
  }
}

/// Errors for model downloading
public enum TextEncoderModelDownloaderError: LocalizedError {
  case modelNotFound
  case qwen3ModelNotFound
  case downloadFailed(String)
  case invalidToken
  /// Variant has been intentionally cut from CDN provisioning. The follow-up
  /// CDN mission tracked under `docs/missions/` will re-enable it.
  case notProvisionedOnCDN(variant: ModelVariant)

  public var errorDescription: String? {
    switch self {
    case .modelNotFound:
      return "Model not found"
    case .qwen3ModelNotFound:
      return "Qwen3 model not found"
    case .downloadFailed(let reason):
      return "Download failed: \(reason)"
    case .invalidToken:
      return "Invalid auth token"
    case .notProvisionedOnCDN(let variant):
      return
        "Variant \(variant.shortName) is not provisioned on the Acervo CDN. "
        + "It will be re-enabled in a follow-up CDN mission."
    }
  }
}
