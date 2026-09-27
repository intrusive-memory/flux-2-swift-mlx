// IncrementalQuantizationTests.swift - chunked on-the-fly quantization
//
// Guards the fix for the GPU-watchdog crash in the iPad-16GB qint8 smoke job:
// on-the-fly quantization must evaluate one block at a time, and the result
// must be identical to MLXNN's whole-model quantize().

import MLX
import MLXNN
import Testing

@testable import Flux2Core

private final class TinyBlock: Module {
  @ModuleInfo var a: Linear
  @ModuleInfo var b: Linear
  @ModuleInfo var norm: LayerNorm

  init(dim: Int) {
    self._a.wrappedValue = Linear(dim, dim)
    self._b.wrappedValue = Linear(dim, dim, bias: false)
    self._norm.wrappedValue = LayerNorm(dimensions: dim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    b(norm(a(x)))
  }
}

private final class TinyModel: Module {
  @ModuleInfo var inProj: Linear
  let blocks: [TinyBlock]
  @ModuleInfo var outProj: Linear

  init(dim: Int = 64, blockCount: Int = 2) {
    self._inProj.wrappedValue = Linear(dim, dim)
    self.blocks = (0..<blockCount).map { _ in TinyBlock(dim: dim) }
    self._outProj.wrappedValue = Linear(dim, dim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = inProj(x)
    for block in blocks { h = block(h) }
    return outProj(h)
  }
}

/// Array elements that are bare `Linear`s have no sub-root to quantize, so
/// they take the deferred (all-at-once) path.
private final class BareLinearArrayModel: Module {
  @ModuleInfo var layers: [Linear]

  init(dim: Int = 64, count: Int = 3) {
    self._layers.wrappedValue = (0..<count).map { _ in Linear(dim, dim) }
  }
}

@Suite("IncrementalQuantization")
struct IncrementalQuantizationTests {

  @Test(arguments: [
    ("transformerBlocks.3.attn.toQ.weight", "transformerBlocks.3"),
    ("transformerBlocks.3.attn.toQ", "transformerBlocks.3"),
    ("singleTransformerBlocks.17.attn.toQkvMlp.scales", "singleTransformerBlocks.17"),
    ("timeGuidanceEmbed.timestepEmbedder.linear1", "timeGuidanceEmbed"),
    ("xEmbedder.weight", "xEmbedder"),
    ("projOut", "projOut"),
  ])
  func chunkKeyGroupsByBlock(path: String, expected: String) {
    #expect(IncrementalQuantization.chunkKey(for: path) == expected)
  }

  @Test func quantizesOneChunkPerBlockAndTopLevelLayer() {
    let model = TinyModel(blockCount: 3)
    eval(model.parameters())

    var chunksEvaluated = 0
    let chunks = IncrementalQuantization.quantize(
      model: model, groupSize: 64, bits: 8, afterChunk: { chunksEvaluated += 1 })

    #expect(
      Set(chunks) == ["inProj", "blocks.0", "blocks.1", "blocks.2", "outProj"])
    #expect(chunks.count == 5)
    #expect(chunksEvaluated == chunks.count)

    // Every Linear became a QuantizedLinear; the LayerNorms were left alone.
    let leaves = model.leafModules().flattened()
    #expect(leaves.filter { $0.1 is QuantizedLinear }.count == 2 + 3 * 2)
    #expect(!leaves.contains { $0.1 is Linear && !($0.1 is QuantizedLinear) })
    #expect(leaves.filter { $0.1 is LayerNorm }.count == 3)
  }

  @Test func matchesWholeModelQuantize() {
    MLXRandom.seed(1234)
    let incremental = TinyModel()
    eval(incremental.parameters())

    let reference = TinyModel()
    reference.update(parameters: incremental.parameters())
    eval(reference.parameters())

    IncrementalQuantization.quantize(model: incremental, groupSize: 64, bits: 8)
    MLXNN.quantize(model: reference, groupSize: 64, bits: 8)
    eval(reference.parameters())

    let x = MLXRandom.normal([2, 64])
    let diff = abs(incremental(x) - reference(x)).max().item(Float.self)
    #expect(diff == 0)
  }

  @Test func alreadyQuantizedModelIsANoOp() {
    let model = TinyModel()
    MLXNN.quantize(model: model, groupSize: 64, bits: 8)
    eval(model.parameters())

    var chunksEvaluated = 0
    let chunks = IncrementalQuantization.quantize(
      model: model, groupSize: 64, bits: 8, afterChunk: { chunksEvaluated += 1 })

    #expect(chunks.isEmpty)
    #expect(chunksEvaluated == 0)
  }

  @Test func bareLinearArrayElementsAreQuantizedTogether() {
    let model = BareLinearArrayModel()
    eval(model.parameters())

    var chunksEvaluated = 0
    let chunks = IncrementalQuantization.quantize(
      model: model, groupSize: 64, bits: 8, afterChunk: { chunksEvaluated += 1 })

    #expect(Set(chunks) == ["layers.0", "layers.1", "layers.2"])
    #expect(chunksEvaluated == 1)
    #expect(model.layers.count == 3)
    #expect(model.leafModules().flattened().allSatisfy { $0.1 is QuantizedLinear })
  }

  /// The production call site: a real (tiny-config) Flux.2 transformer gets
  /// one chunk per double/single block, and every Linear ends up quantized.
  @Test func flux2TransformerIsQuantizedBlockByBlock() {
    let config = Flux2TransformerConfig(
      numLayers: 2, numSingleLayers: 3, attentionHeadDim: 128, numAttentionHeads: 2,
      jointAttentionDim: 256)
    let transformer = Flux2Transformer2DModel(config: config)
    eval(transformer.parameters())

    let chunks = IncrementalQuantization.quantize(
      model: transformer, groupSize: 64, bits: 8)

    let blockChunks = chunks.filter { $0.contains(".") }
    #expect(
      Set(blockChunks) == [
        "transformerBlocks.0", "transformerBlocks.1",
        "singleTransformerBlocks.0", "singleTransformerBlocks.1", "singleTransformerBlocks.2",
      ])
    let leaves = transformer.leafModules().flattened()
    #expect(!leaves.contains { $0.1 is Linear && !($0.1 is QuantizedLinear) })
    #expect(leaves.contains { $0.1 is QuantizedLinear })
  }
}
