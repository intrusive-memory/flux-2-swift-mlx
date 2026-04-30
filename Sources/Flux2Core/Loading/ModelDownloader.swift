// ModelDownloader.swift - Download Flux.2 models via SwiftAcervo CDN
// Copyright 2025 Vincent Gourbin
//
// Sortie 19 (OPERATION FAREWELL EMBRACE) replaced the hand-rolled HuggingFace
// API client (HF tree-listing fetcher, per-file resolve downloads, and the
// legacy `~/.cache/huggingface/hub` fallback) with `Acervo.ensureAvailable(...)`
// from SwiftAcervo. The downloader's public API surface is preserved aside
// from the new `Flux2DownloadError.notProvisionedOnCDN(variant:)` case
// (Tasks 2 + 5).

import Foundation
import SwiftAcervo

/// Progress callback for download updates
public typealias Flux2DownloadProgressCallback = @Sendable (Double, String) -> Void

/// Downloads Flux.2 models via the Acervo CDN.
///
/// Storage location is `Acervo.sharedModelsDirectory` (App Group container or
/// Application Support fallback). The legacy `ModelRegistry.modelsDirectory`
/// path is no longer the canonical location — we forward through Acervo.
public class Flux2ModelDownloader: @unchecked Sendable {

  /// Optional auth token preserved on the type for API stability.
  /// SwiftAcervo CDN reads are unauthenticated; this field is retained so
  /// `init(hfToken:)` does not break consumers (legacy callers in
  /// `ModelManager.swift` still surface it from `HF_TOKEN` env / UserDefaults).
  private var hfToken: String?  // swift-format-ignore: legacy name preserved

  public init(hfToken: String? = nil) {
    self.hfToken = hfToken
    if let token = hfToken {
      setenv("HF_TOKEN", token, 1)
    }
  }

  // MARK: - Acervo Model IDs (formerly `huggingFaceRepo`)
  //
  // SwiftAcervo addresses content by the same "org/repo" string format as
  // HuggingFace (slugified internally by `Acervo.slugify(_:)`). The accessor
  // `huggingFaceRepo` on `TransformerVariant` / `VAEVariant` is the source of
  // truth for these IDs; Sortie 20 renames that accessor to `repoId`. Until
  // then we read it under its current name.

  /// Acervo model ID for a component.
  ///
  /// The returned string is an Acervo model ID (slugified internally by
  /// SwiftAcervo when used as a directory name). It currently equals the
  /// HuggingFace `<org>/<repo>` form because Acervo's CDN ships use the same
  /// addressing scheme.
  private static func repoId(for component: ModelRegistry.ModelComponent) -> String {
    switch component {
    case .transformer(let variant):
      return variant.huggingFaceRepo
    case .textEncoder:
      // Text encoder downloads are owned by FluxTextEncoders.TextEncoderModelDownloader.
      // Returned for reference only; this class does not download text encoders.
      return "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    case .vae:
      // VAE ships as part of the Klein-4B repo (NOT gated, and the same VAE
      // weights apply to every Flux.2 model).
      return "black-forest-labs/FLUX.2-klein-4B"
    }
  }

  /// Subfolder relative to the Acervo model directory where this component's
  /// safetensors actually live. The Pipeline + VAE loaders expect a directory
  /// containing the safetensors at depth 0 (`contentsOfDirectory` is non-recursive),
  /// so we must return the right depth for each variant's manifest layout.
  private static func subfolder(for component: ModelRegistry.ModelComponent) -> String? {
    switch component {
    case .transformer(let variant):
      return variant.huggingFaceSubfolder
    case .vae:
      return "vae"
    case .textEncoder:
      return nil
    }
  }

  // MARK: - Model Paths

  /// Check if a model component is downloaded
  public static func isDownloaded(_ component: ModelRegistry.ModelComponent) -> Bool {
    findModelPath(for: component) != nil
  }

  /// Find local path for a model component within the Acervo shared models
  /// directory. Returns the directory containing the component's safetensors
  /// (with the subfolder appended for components like VAE / qint8) only when
  /// the local files are present and (for sharded models) complete.
  ///
  /// Note: We do not use `Acervo.isModelAvailable(_:)` here because that
  /// helper only probes for `config.json` at the model root. Subfolder-shipped
  /// repos (e.g., the qint8 transformer at `flux-2-dev/transformer/qint8/`)
  /// would falsely report unavailable. We instead resolve the model directory
  /// and probe for the expected layout directly.
  public static func findModelPath(for component: ModelRegistry.ModelComponent) -> URL? {
    let repoId = repoId(for: component)
    guard let modelDir = try? Acervo.modelDirectory(for: repoId),
      FileManager.default.fileExists(atPath: modelDir.path)
    else {
      return nil
    }
    var path = modelDir
    // Components shipped under a subfolder need the suffix so that
    // `contentsOfDirectory(atPath:)` on the returned URL sees the safetensors
    // at depth 0 (the weight loaders are non-recursive).
    if let sub = subfolder(for: component) {
      let subPath = path.appendingPathComponent(sub)
      if FileManager.default.fileExists(atPath: subPath.path) {
        path = subPath
      }
    }
    // Require either config.json or model_index.json at the resolved depth
    // before declaring the component "found". This mirrors the legacy probe.
    let hasConfig = FileManager.default.fileExists(
      atPath: path.appendingPathComponent("config.json").path)
    let hasModelIndex = FileManager.default.fileExists(
      atPath: path.appendingPathComponent("model_index.json").path)
    guard hasConfig || hasModelIndex else {
      return nil
    }
    let verification = verifyModel(at: path)
    return verification.complete ? path : nil
  }

  /// Verify that a model directory has all required safetensors files.
  ///
  /// Acervo's manifest already verifies SHA-256 + size per file. This check is
  /// retained as a complementary local-side shard-completeness sanity check
  /// (e.g., to catch a partially-deleted directory): index.json is not
  /// trusted; the shard filenames themselves declare totals.
  public static func verifyModel(at path: URL) -> (complete: Bool, missing: [String]) {
    let contents = (try? FileManager.default.contentsOfDirectory(atPath: path.path)) ?? []
    let safetensorsFiles = contents.filter { $0.hasSuffix(".safetensors") }

    // Single file model (various naming conventions)
    if safetensorsFiles.contains("model.safetensors")
      || safetensorsFiles.contains("diffusion_pytorch_model.safetensors")
    {
      return (true, [])
    }

    // Klein bf16 models use flux-2-klein-*.safetensors naming
    if safetensorsFiles.contains(where: { $0.hasPrefix("flux-2-klein") }) {
      return (true, [])
    }

    // Check for sharded model
    guard !safetensorsFiles.isEmpty else {
      return (false, ["No safetensors files found"])
    }

    // Parse sharded pattern: model-XXXXX-of-YYYYY.safetensors
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

    // Has some safetensors files but not in standard sharded format — treat as
    // a custom layout (e.g., quanto qint8 ships) and trust the manifest.
    return (true, [])
  }

  /// Total on-disk size of a downloaded model component in bytes.
  ///
  /// Returns `nil` if the component is not found on disk. Recursively
  /// enumerates all files under the component directory so that multi-file
  /// safetensors shards and ancillary JSON files are all counted.
  public static func diskSize(for component: ModelRegistry.ModelComponent) -> Int64? {
    guard let path = findModelPath(for: component) else { return nil }
    let fm = FileManager.default
    guard
      let enumerator = fm.enumerator(
        at: path, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
    else {
      return nil
    }
    var total: Int64 = 0
    for case let fileURL as URL in enumerator {
      guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
        values.isRegularFile == true,
        let size = values.fileSize
      else { continue }
      total += Int64(size)
    }
    return total
  }

  // MARK: - Download

  /// Download a model component via the Acervo CDN.
  ///
  /// Throws `Flux2DownloadError.notProvisionedOnCDN(variant:)` for transformer
  /// variants intentionally cut from CDN provisioning (Sortie 19, Task 2):
  /// `bf16`, `klein4B_base_bf16`, `klein9B_base_bf16`, `klein9B_kv_bf16`. UI
  /// callers that iterate `TransformerVariant.allCases` can pre-filter via
  /// `TransformerVariant.isProvisionedOnCDN` to avoid this error path.
  public func download(
    _ component: ModelRegistry.ModelComponent,
    progress: Flux2DownloadProgressCallback? = nil
  ) async throws -> URL {
    // Refuse cut variants up front.
    if case .transformer(let variant) = component, !variant.isProvisionedOnCDN {
      throw Flux2DownloadError.notProvisionedOnCDN(variant: variant)
    }

    // Fast path: already present.
    if let existingPath = Self.findModelPath(for: component) {
      progress?(1.0, "Model already downloaded")
      return existingPath
    }

    let repoId = Self.repoId(for: component)
    progress?(0.0, "Starting download of \(component.displayName)...")
    Flux2Debug.log("Downloading \(component.displayName) via Acervo CDN: \(repoId)")

    // Empty `files: []` instructs Acervo to download every file in the CDN
    // manifest. The manifest is the authoritative source of truth and was
    // shipped per WU1 (Sorties 5 / 7 / 11 / 12). For the qint8 subfolder ship
    // (Sortie 11), the manifest only contains the qint8 files, so no runtime
    // filter is needed. For the full-repo ships (Klein-4B, Klein-9B,
    // Klein-4B-int8) every file in the repo is shipped — the VAE component
    // shares the Klein-4B model directory, so requesting it after the
    // transformer is a no-op (`isModelAvailable` short-circuits).
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

    guard let modelPath = Self.findModelPath(for: component) else {
      throw Flux2DownloadError.downloadFailed(
        "Acervo download succeeded but local files for \(component.displayName) "
          + "could not be located under \(repoId).")
    }

    let verification = Self.verifyModel(at: modelPath)
    if !verification.complete {
      Flux2Debug.log(
        "Warning: local shard verification reports missing files for "
          + "\(component.displayName): \(verification.missing)")
    }

    progress?(1.0, "Download complete")
    return modelPath
  }

  /// Download all models for a quantization configuration
  public func downloadAll(
    for config: Flux2QuantizationConfig,
    progress: Flux2DownloadProgressCallback? = nil
  ) async throws {
    let components: [ModelRegistry.ModelComponent] = [
      .transformer(ModelRegistry.TransformerVariant(rawValue: config.transformer.rawValue)!),
      .vae(.standard),
    ]

    let totalComponents = Float(components.count + 1)  // +1 for text encoder

    // Download transformer and VAE
    for (index, component) in components.enumerated() {
      let completedComponents = Float(index)
      let componentProgress: Flux2DownloadProgressCallback = { p, msg in
        let overall = (completedComponents + Float(p)) / totalComponents
        progress?(Double(overall), msg)
      }

      _ = try await download(component, progress: componentProgress)
    }

    // Text encoder is handled by FluxTextEncoders / TextEncoderModelDownloader.
    progress?(1.0, "All models downloaded")
  }

  // MARK: - Utilities

  /// Format bytes as human-readable string
  public static func formatSize(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.allowedUnits = [.useGB, .useMB]
    formatter.countStyle = .file
    return formatter.string(fromByteCount: bytes)
  }

  /// Delete a downloaded model. Idempotent: returns silently if the model is
  /// not present locally.
  ///
  /// Important: the Acervo model directory may host more than one logical
  /// component — e.g., `vae(.standard)` and `transformer(.klein4B_bf16)`
  /// both ship under `black-forest-labs/FLUX.2-klein-4B`. Deleting via the
  /// component-level entry point removes the entire shared directory; UI
  /// surfaces that distinguish "delete VAE" from "delete Klein-4B transformer"
  /// will need additional logic in a follow-up sortie.
  public static func delete(_ component: ModelRegistry.ModelComponent) throws {
    let repoId = repoId(for: component)
    guard let modelDir = try? Acervo.modelDirectory(for: repoId),
      FileManager.default.fileExists(atPath: modelDir.path)
    else {
      return
    }
    do {
      try Acervo.deleteModel(repoId)
      Flux2Debug.log("Deleted \(component.displayName)")
    } catch {
      Flux2Debug.log(
        "Acervo.deleteModel(\(repoId)) raised: \(error.localizedDescription) — treating as deleted")
    }
  }

  /// Get total size of downloaded models (transformer + VAE).
  ///
  /// Iterates only the variants that are provisioned on the CDN; cut variants
  /// would never have been downloaded by this class so we skip them.
  public static func downloadedSize() -> Int64 {
    var total: Int64 = 0
    var visitedRepoIds: Set<String> = []

    let provisionedTransformerVariants =
      ModelRegistry.TransformerVariant.allCases.filter(\.isProvisionedOnCDN)
    var components: [ModelRegistry.ModelComponent] = provisionedTransformerVariants.map {
      .transformer($0)
    }
    components.append(.vae(.standard))

    for component in components {
      let repoId = repoId(for: component)
      // Multiple components (e.g., klein4B_bf16 + vae) can share an Acervo
      // model directory. Count each directory once.
      if !visitedRepoIds.insert(repoId).inserted {
        continue
      }
      if let path = findModelPath(for: component) {
        total += directorySize(at: path)
      }
    }

    return total
  }

  private static func directorySize(at url: URL) -> Int64 {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
      return 0
    }

    var total: Int64 = 0
    for case let fileURL as URL in enumerator {
      if let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
        let size = attrs[.size] as? Int64
      {
        total += size
      }
    }

    return total
  }
}

// MARK: - TransformerVariant CDN provisioning

extension ModelRegistry.TransformerVariant {
  /// Whether this variant is currently provisioned on the Acervo CDN.
  ///
  /// Returns `false` for the four cut variants (Sortie 19, Task 3):
  /// - `.bf16` — 64 GB FLUX.2-dev BF16; qint8 is the canonical production
  ///   quantization, license redistribution complications cut bf16.
  /// - `.klein4B_base_bf16` — base (non-distilled) for LoRA training, deferred
  ///   until LoRA-training is a v1 surface.
  /// - `.klein9B_base_bf16` — base (non-distilled), LoRA-training only,
  ///   gated.
  /// - `.klein9B_kv_bf16` — multi-reference I2I specialty variant, gated.
  ///
  /// UI surfaces iterating `allCases` should consult this property to gray
  /// out (or hide) variants that would otherwise throw
  /// `Flux2DownloadError.notProvisionedOnCDN(variant:)` from
  /// `Flux2ModelDownloader.download(_:progress:)`.
  public var isProvisionedOnCDN: Bool {
    switch self {
    case .bf16, .klein4B_base_bf16, .klein9B_base_bf16, .klein9B_kv_bf16:
      return false
    case .qint8, .klein4B_bf16, .klein4B_8bit, .klein9B_bf16:
      return true
    }
  }
}

// MARK: - Errors

public enum Flux2DownloadError: LocalizedError {
  case modelNotFound(String)
  case downloadFailed(String)
  case verificationFailed([String])
  case insufficientSpace(required: Int64, available: Int64)
  /// Variant has been intentionally cut from CDN provisioning. The follow-up
  /// CDN provisioning mission tracked under `docs/missions/` will re-enable
  /// it.
  case notProvisionedOnCDN(variant: ModelRegistry.TransformerVariant)

  public var errorDescription: String? {
    switch self {
    case .modelNotFound(let id):
      return "Model not found: \(id)"
    case .downloadFailed(let reason):
      return "Download failed: \(reason)"
    case .verificationFailed(let missing):
      return "Verification failed, missing files: \(missing.joined(separator: ", "))"
    case .insufficientSpace(let required, let available):
      return
        "Insufficient disk space: need \(Flux2ModelDownloader.formatSize(required)), have \(Flux2ModelDownloader.formatSize(available))"
    case .notProvisionedOnCDN(let variant):
      return
        "Variant \(variant.rawValue) is not yet provisioned on the Acervo CDN. "
        + "Track follow-up CDN provisioning mission in docs/missions/."
    }
  }
}
