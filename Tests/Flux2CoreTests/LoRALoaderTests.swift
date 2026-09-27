// LoRALoaderTests.swift - Key-format detection and layer mapping for LoRA files.
//
// A wrong key mapping doesn't crash — it silently produces a LoRA that patches
// nothing. These tests write tiny synthetic safetensors files (rank 2, a few
// floats per tensor) and drive them through the public `load()` entry point,
// asserting on the resulting Swift module paths, QKV splits, target-model
// detection and metadata scale.

import Foundation
import MLX
import Testing

@testable import Flux2Core

@Suite final class LoRALoaderTests {

  let tempDir: URL
  let rank = 2

  init() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("LoRALoaderTests_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  deinit {
    try? FileManager.default.removeItem(at: tempDir)
  }

  /// Build a lora_A/lora_B pair for every base path and save as safetensors.
  /// `outDims` optionally overrides loraB's output dim for a given base path.
  private func loadLoRA(
    basePaths: [String],
    outDims: [String: Int] = [:],
    extraKeys: [String] = [],
    metadata: [String: String] = [:]
  ) throws -> LoRALoader {
    var arrays: [String: MLXArray] = [:]
    for base in basePaths {
      arrays["\(base).lora_A.weight"] = MLXArray.zeros([rank, 4])
      arrays["\(base).lora_B.weight"] = MLXArray.zeros([outDims[base] ?? 4, rank])
    }
    for key in extraKeys {
      arrays[key] = MLXArray.zeros([1])
    }
    let url = tempDir.appendingPathComponent("lora_\(UUID().uuidString).safetensors")
    try MLX.save(arrays: arrays, metadata: metadata, url: url)

    let loader = LoRALoader(config: LoRAConfig(filePath: url.path))
    try loader.load()
    return loader
  }

  // MARK: - Errors

  @Test func missingFileThrowsFileNotFound() {
    let path = tempDir.appendingPathComponent("nope.safetensors").path
    let loader = LoRALoader(config: LoRAConfig(filePath: path))
    #expect {
      try loader.load()
    } throws: { error in
      guard case LoRALoaderError.fileNotFound(let p) = error else { return false }
      return p == path
    }
  }

  // MARK: - Diffusers format

  @Test func diffusersKeysMapToSwiftModulePaths() throws {
    let loader = try loadLoRA(basePaths: [
      "transformer.transformer_blocks.0.attn.to_q",
      "transformer.transformer_blocks.0.attn.to_k",
      "transformer.transformer_blocks.0.attn.to_v",
      "transformer.transformer_blocks.0.attn.to_out.0",
      "transformer.transformer_blocks.0.attn.add_q_proj",
      "transformer.transformer_blocks.0.attn.to_add_out",
      "transformer.transformer_blocks.0.ff.linear_out",
      "transformer.transformer_blocks.0.ff_context.linear_out",
      "transformer.single_transformer_blocks.3.attn.to_qkv_mlp_proj",
      "transformer.single_transformer_blocks.3.attn.to_out",
      "transformer.x_embedder",
      "transformer.context_embedder",
      "transformer.proj_out",
      "transformer.norm_out.linear",
      "transformer.time_guidance_embed.timestep_embedder.linear_1",
      "transformer.double_stream_modulation_img.linear",
    ])

    #expect(
      Set(loader.layerPaths) == [
        "transformerBlocks.0.attn.toQ",
        "transformerBlocks.0.attn.toK",
        "transformerBlocks.0.attn.toV",
        "transformerBlocks.0.attn.toOut",
        "transformerBlocks.0.attn.addQProj",
        "transformerBlocks.0.attn.toAddOut",
        "transformerBlocks.0.ff.linearOut",
        "transformerBlocks.0.ffContext.linearOut",
        "singleTransformerBlocks.3.attn.toQkvMlp",
        "singleTransformerBlocks.3.attn.toOut",
        "xEmbedder",
        "contextEmbedder",
        "projOut",
        "normOut.linear",
        "timeGuidanceEmbed.timestepEmbedder.linear1",
        "doubleStreamModulationImg.linear",
      ])
    #expect(loader.info?.numLayers == 16)
    #expect(loader.info?.rank == rank)
  }

  @Test func baseModelPrefixIsStripped() throws {
    let loader = try loadLoRA(basePaths: [
      "base_model.model.transformer_blocks.1.attn.to_q"
    ])
    #expect(loader.layerPaths == ["transformerBlocks.1.attn.toQ"])
  }

  @Test func nonLoRAAndUnpairedKeysAreIgnored() throws {
    var arrays: [String: MLXArray] = [
      "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": MLXArray.zeros([rank, 4]),
      "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": MLXArray.zeros([4, rank]),
      // lora_A without its lora_B partner
      "transformer.transformer_blocks.0.attn.to_k.lora_A.weight": MLXArray.zeros([rank, 4]),
      // unrelated tensors
      "transformer.transformer_blocks.0.attn.to_q.alpha": MLXArray.zeros([1]),
    ]
    arrays["__internal.x.lora_A.weight"] = MLXArray.zeros([rank, 4])
    let url = tempDir.appendingPathComponent("partial.safetensors")
    try MLX.save(arrays: arrays, url: url)

    let loader = LoRALoader(config: LoRAConfig(filePath: url.path))
    try loader.load()

    #expect(loader.layerPaths == ["transformerBlocks.0.attn.toQ"])
    #expect(loader.info?.numLayers == 1)
    // numParameters counts both matrices of the one complete pair.
    #expect(loader.info?.numParameters == rank * 4 + 4 * rank)
  }

  // MARK: - BFL format

  @Test func bflKeysMapToSwiftModulePaths() throws {
    let loader = try loadLoRA(basePaths: [
      "diffusion_model.double_blocks.2.img_attn.proj",
      "diffusion_model.double_blocks.2.txt_attn.proj",
      "diffusion_model.double_blocks.2.img_mlp.0",
      "diffusion_model.double_blocks.2.img_mlp.2",
      "diffusion_model.double_blocks.2.txt_mlp.0",
      "diffusion_model.double_blocks.2.txt_mlp.2",
      "diffusion_model.single_blocks.11.linear1",
      "diffusion_model.single_blocks.11.linear2",
      "diffusion_model.double_stream_modulation_txt.lin",
      "diffusion_model.single_stream_modulation.lin",
      "diffusion_model.img_in",
      "diffusion_model.txt_in",
      "diffusion_model.final_layer.linear",
      "diffusion_model.time_in.in_layer",
    ])

    #expect(
      Set(loader.layerPaths) == [
        "transformerBlocks.2.attn.toOut",
        "transformerBlocks.2.attn.toAddOut",
        "transformerBlocks.2.ff.activation.proj",
        "transformerBlocks.2.ff.linearOut",
        "transformerBlocks.2.ffContext.activation.proj",
        "transformerBlocks.2.ffContext.linearOut",
        "singleTransformerBlocks.11.attn.toQkvMlp",
        "singleTransformerBlocks.11.attn.toOut",
        "doubleStreamModulationTxt.linear",
        "singleStreamModulation.linear",
        "xEmbedder",
        "contextEmbedder",
        "projOut",
        "timeGuidanceEmbed.timestepEmbedder.inLayer",
      ])
  }

  @Test func bflCombinedQKVIsSplitIntoThreeProjections() throws {
    let imgQKV = "double_blocks.7.img_attn.qkv"
    let txtQKV = "double_blocks.7.txt_attn.qkv"
    let loader = try loadLoRA(
      basePaths: [imgQKV, txtQKV],
      outDims: [imgQKV: 12, txtQKV: 12])

    #expect(
      Set(loader.layerPaths) == [
        "transformerBlocks.7.attn.toQ",
        "transformerBlocks.7.attn.toK",
        "transformerBlocks.7.attn.toV",
        "transformerBlocks.7.attn.addQProj",
        "transformerBlocks.7.attn.addKProj",
        "transformerBlocks.7.attn.addVProj",
      ])

    // loraB [12, rank] is split into three [4, rank] slices; loraA is shared.
    for path in loader.layerPaths {
      let pair = try #require(loader.getWeights(for: path))
      #expect(pair.loraB.shape == [4, rank])
      #expect(pair.loraA.shape == [rank, 4])
      #expect(pair.rank == rank)
    }
    #expect(loader.info?.numLayers == 6)
  }

  @Test func unmappedBFLPathIsKeptVerbatim() throws {
    let loader = try loadLoRA(basePaths: ["double_blocks.0.some_new_layer"])
    #expect(loader.layerPaths == ["double_blocks.0.some_new_layer"])
  }

  // MARK: - Target model detection

  /// Emit one key per block index so detection sees max indices.
  private func blockPaths(double: Int, single: Int, diffusers: Bool) -> [String] {
    let d =
      (0..<double).map {
        diffusers ? "transformer_blocks.\($0).attn.to_q" : "double_blocks.\($0).img_attn.proj"
      }
    let s =
      (0..<single).map {
        diffusers
          ? "single_transformer_blocks.\($0).attn.to_out" : "single_blocks.\($0).linear2"
      }
    return d + s
  }

  @Test(
    arguments: [
      (5, 20, LoRAInfo.TargetModel.klein4B),
      (8, 24, .klein9B),
      (8, 48, .dev),
      (3, 10, .unknown),
    ])
  func detectsTargetModelFromBlockCounts(
    double: Int, single: Int, expected: LoRAInfo.TargetModel
  ) throws {
    for diffusers in [true, false] {
      let loader = try loadLoRA(
        basePaths: blockPaths(double: double, single: single, diffusers: diffusers))
      #expect(
        loader.info?.targetModel == expected,
        "diffusers=\(diffusers) \(double)x\(single)")
    }
  }

  @Test func multiDigitBlockIndicesAreParsed() throws {
    // Only the highest indices matter for detection: 4 and 19 → Klein 4B.
    let loader = try loadLoRA(basePaths: [
      "transformer_blocks.4.attn.to_q",
      "single_transformer_blocks.19.attn.to_out",
    ])
    #expect(loader.info?.targetModel == .klein4B)
    #expect(loader.layerPaths.contains("singleTransformerBlocks.19.attn.toOut"))
  }

  // MARK: - Metadata scale

  @Test func metadataAlphaOverRankSetsScale() throws {
    let loader = try loadLoRA(
      basePaths: ["transformer_blocks.0.attn.to_q"],
      metadata: ["lora_alpha": "8", "lora_rank": "16"])
    #expect(loader.metadataScale == 0.5)
  }

  @Test(arguments: [
    [:],
    ["lora_alpha": "8"],
    ["lora_alpha": "8", "lora_rank": "0"],
    ["lora_alpha": "x", "lora_rank": "4"],
  ])
  func missingOrInvalidMetadataDefaultsScaleToOne(metadata: [String: String]) throws {
    let loader = try loadLoRA(
      basePaths: ["transformer_blocks.0.attn.to_q"], metadata: metadata)
    #expect(loader.metadataScale == 1.0)
  }

  // MARK: - Fusion cleanup

  @Test func clearWeightsAfterFusionDropsWeightsButKeepsInfo() throws {
    let loader = try loadLoRA(basePaths: ["transformer_blocks.0.attn.to_q"])
    #expect(loader.hasWeightsInMemory)

    loader.clearWeightsAfterFusion()

    #expect(!loader.hasWeightsInMemory)
    #expect(loader.layerPaths.isEmpty)
    #expect(loader.getWeights(for: "transformerBlocks.0.attn.toQ") == nil)
    #expect(loader.info?.numLayers == 1)
  }
}
