// IncrementalQuantization.swift - Bounded, chunked on-the-fly quantization
// Copyright 2025 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// On-the-fly quantization that evaluates the model one chunk at a time.
///
/// `quantize(model:)` from MLXNN only rewires the module tree: every
/// `Linear` → `QuantizedLinear` conversion is a *lazy* `MLX.quantized(weight)`
/// node whose input is the (also lazy, disk-backed) bf16 weight. Following it
/// with a single `eval(model.parameters())` materializes the entire model —
/// for Klein 4B that is ~8 GB of bf16 page-ins plus ~200 quantize kernels —
/// in one evaluation.
///
/// On hosts with little headroom (a 7 GB GitHub-hosted `macos-26` runner with a
/// paravirtualized GPU, or a real 8–16 GB iPad) that one evaluation keeps GPU
/// command buffers in flight while the OS pages multi-GB of weights in under
/// memory pressure, which trips the Metal watchdog
/// (`kIOGPUCommandBufferCallbackErrorTimeout`) and aborts the process.
///
/// Quantizing and evaluating one transformer block (or top-level layer) at a
/// time bounds both the amount of work per command buffer and the bf16
/// working set: each chunk's bf16 source weights become unreachable as soon
/// as its packed weights are evaluated, and the MLX buffer cache is released
/// before the next chunk starts.
enum IncrementalQuantization {

  /// The chunk a leaf-module or parameter path belongs to.
  ///
  /// Paths are grouped up to and including the first numeric component, so
  /// `transformerBlocks.3.attn.toQ.weight` → `transformerBlocks.3`. Paths with
  /// no numeric component are grouped by their first component, so
  /// `timeGuidanceEmbed.timestepEmbedder.linear1` → `timeGuidanceEmbed`.
  static func chunkKey(for path: String) -> String {
    let components = path.split(separator: ".", omittingEmptySubsequences: false)
    if let index = components.firstIndex(where: { Int($0) != nil }) {
      return components[...index].joined(separator: ".")
    }
    return components.first.map(String.init) ?? path
  }

  /// Quantize every quantizable leaf of `model`, evaluating chunk by chunk.
  ///
  /// Produces the same module tree as
  /// `quantize(model:groupSize:bits:)` followed by `eval(model.parameters())`.
  ///
  /// - Parameters:
  ///   - model: The module to quantize in place.
  ///   - groupSize: Quantization group size.
  ///   - bits: Bits per weight.
  ///   - afterChunk: Called after each chunk is evaluated (before the next one
  ///     starts). Defaults to releasing the MLX buffer cache.
  /// - Returns: The chunk keys, in the order they were quantized.
  @discardableResult
  static func quantize(
    model: Module,
    groupSize: Int,
    bits: Int,
    afterChunk: () -> Void = { Memory.clearCache() }
  ) -> [String] {
    var chunkOrder: [String] = []
    var seen = Set<String>()
    for (path, module) in model.leafModules().flattened()
    where module is Quantizable && !(module is Quantized) {
      let key = chunkKey(for: path)
      if seen.insert(key).inserted {
        chunkOrder.append(key)
      }
    }

    // `Module.update(modules:)` rejects sparse arrays, so a chunk inside an
    // array (`transformerBlocks.3`) is quantized by passing that block itself
    // as the root. Top-level chunks (`xEmbedder`, `timeGuidanceEmbed`) contain
    // no arrays on their path, so they are safe to update from the model root.
    //
    // Only the (non-leaf) chunk roots are retained here. Holding on to every
    // module (e.g. a `namedModules()` dictionary) would keep each replaced
    // bf16 `Linear` — and its weight buffer — alive until the whole pass
    // finishes, defeating the point of chunking.
    let chunkKeys = Set(chunkOrder)
    var chunkRoots: [String: Module] = [:]
    for (path, module) in model.namedModules()
    where chunkKeys.contains(path) && !(module is Quantizable) {
      chunkRoots[path] = module
    }
    var deferred: [String] = []

    for key in chunkOrder {
      let isArrayElement = key.split(separator: ".").contains { Int($0) != nil }
      if isArrayElement {
        guard let chunkModule = chunkRoots.removeValue(forKey: key) else {
          // An array whose elements are themselves bare Linear layers: no
          // root that can be updated without a sparse array. Quantize these
          // together at the end (the pre-chunking behavior).
          deferred.append(key)
          continue
        }
        MLXNN.quantize(model: chunkModule, groupSize: groupSize, bits: bits)
      } else {
        MLXNN.quantize(
          model: model, groupSize: groupSize, bits: bits,
          filter: { path, _ in chunkKey(for: path) == key })
      }
      evaluate(chunks: [key], of: model)
      afterChunk()
    }

    if !deferred.isEmpty {
      let deferredKeys = Set(deferred)
      MLXNN.quantize(
        model: model, groupSize: groupSize, bits: bits,
        filter: { path, _ in deferredKeys.contains(chunkKey(for: path)) })
      evaluate(chunks: deferredKeys, of: model)
      afterChunk()
    }

    return chunkOrder
  }

  private static func evaluate<Keys: Sequence<String>>(chunks keys: Keys, of model: Module) {
    let keySet = Set(keys)
    let chunkParameters = model.parameters().flattened()
      .filter { keySet.contains(chunkKey(for: $0.0)) }
      .map(\.1)
    eval(chunkParameters)
  }
}
