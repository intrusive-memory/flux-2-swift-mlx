// Flux2CoreModuleTests.swift — MLX-runtime tests for Flux2Core internal modules.
//
// These tests instantiate transformer/attention modules and run forward passes that
// require Metal eval. They previously lived in Flux2CoreTests with `if isCI { return }`
// guards that bailed *after* allocating modules and random tensors, paying the setup
// cost on CI for tests that did nothing useful. Moved here per TESTING_REQUIREMENTS §5.

import Testing
import Foundation
@testable import Flux2Core
import MLX

// MARK: - Embedding Tests

@Suite struct EmbeddingTests {

    @Test func timestepEmbedding() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let embedder = Flux2TimestepGuidanceEmbeddings(
            embeddingDim: 256,
            timeEmbedDim: 6144,
            useGuidanceEmbeds: true
        )

        let timestep = MLXArray([Float(0.5)])
        let guidance = MLXArray([Float(4.0)])

        let embedding = embedder(timestep: timestep, guidance: guidance)

        #expect(embedding.shape == [1, 6144])
    }

    @Test func roPE() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let rope = Flux2RoPE(axesDims: [32, 32, 32, 32], theta: 2000.0)

        // Create position IDs: [100, 4]
        var flatData: [Int32] = []
        for i: Int32 in 0..<100 {
            flatData.append(contentsOf: [Int32(0), i / 10, i % 10, Int32(0)])
        }
        let ids = MLXArray(flatData).reshaped([100, 4])

        let (cosEmb, sinEmb) = rope(ids)

        // Should output [100, 128] (sum of axes dims)
        #expect(cosEmb.shape[0] == 100)
        #expect(sinEmb.shape[0] == 100)
    }
}

// MARK: - Integration Tests

@Suite struct IntegrationTests {

    @Test func modulationFlow() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let dim = 6144
        let modulation = Flux2Modulation(dim: dim, numSets: 2)

        let embedding = MLXRandom.normal([1, dim])
        let params = modulation(embedding)

        #expect(params.count == 2)

        for param in params {
            #expect(param.shift.shape == [1, dim])
            #expect(param.scale.shape == [1, dim])
            #expect(param.gate.shape == [1, dim])
        }
    }

    @Test func feedForwardShape() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let dim = 6144
        let ff = Flux2FeedForward(dim: dim)

        let input = MLXRandom.normal([1, 100, dim])
        let output = ff(input)

        #expect(output.shape == input.shape)
    }
}

// MARK: - Weight Conversion Tests (bf16/f16 dtype consistency)

@Suite struct WeightConversionTests {

    /// Validate that bf16->f16 direct conversion produces identical results to bf16->f32->f16.
    /// This is critical for the BFL weight loading optimization (item 2.1).
    @Test func bf16ToF16DirectMatchesViaF32() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let testValues: [Float] = [
            0.0, 1.0, -1.0,
            0.001, -0.001,           // Small weights
            0.01, -0.05, 0.1,       // Typical weight magnitudes
            0.5, -0.5, 1.5, -2.0,   // Normal range
            100.0, -100.0,           // Larger values
            65504.0, -65504.0,       // f16 max representable
            0.000061035,             // f16 min positive normal (~6.1e-5)
        ]
        let f32Array = MLXArray(testValues)
        let bf16Array = f32Array.asType(.bfloat16)

        let directF16 = bf16Array.asType(.float16)
        let viaF32 = bf16Array.asType(.float32).asType(.float16)

        eval(directF16, viaF32)

        let directValues = directF16.asType(.float32)
        let viaF32Values = viaF32.asType(.float32)

        for i in 0..<testValues.count {
            let d = directValues[i].item(Float.self)
            let v = viaF32Values[i].item(Float.self)
            #expect(d == v, "Mismatch at index \(i) for input \(testValues[i]): direct=\(d), viaF32=\(v)")
        }
    }

    /// Test bf16->f16 with a realistic weight matrix (3072×3072 linear layer).
    @Test func bf16ToF16DirectLargeArray() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let weights = MLXRandom.normal([3072, 3072]).asType(.bfloat16)

        let directF16 = weights.asType(.float16)
        let viaF32 = weights.asType(.float32).asType(.float16)

        eval(directF16, viaF32)

        let diff = MLX.abs(directF16.asType(.float32) - viaF32.asType(.float32))
        let maxDiff = MLX.max(diff).item(Float.self)
        #expect(maxDiff == 0.0, "Max difference between direct and viaF32 conversion: \(maxDiff)")

        let hasNaN = any(isNaN(directF16)).item(Bool.self)
        let hasInf = any(MLX.abs(directF16.asType(.float32)) .> 1e30).item(Bool.self)
        #expect(!hasNaN, "Direct bf16->f16 conversion should not introduce NaN")
        #expect(!hasInf, "Normal weight values should not overflow to inf")
    }

    /// Test that values outside f16 range are handled consistently across both paths.
    @Test func bf16ToF16OverflowConsistency() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let largeValues = MLXArray([Float(70000.0), Float(-70000.0), Float(100000.0)])
            .asType(.bfloat16)

        let directF16 = largeValues.asType(.float16)
        let viaF32 = largeValues.asType(.float32).asType(.float16)

        eval(directF16, viaF32)

        let directF32 = directF16.asType(.float32)
        let viaF32F32 = viaF32.asType(.float32)

        for i in 0..<3 {
            let d = directF32[i].item(Float.self)
            let v = viaF32F32[i].item(Float.self)
            #expect(d == v, "Overflow handling mismatch at index \(i): direct=\(d), viaF32=\(v)")
        }
    }
}

// MARK: - KV Extraction Attention Mask Tests

@Suite struct KVExtractionMaskTests {

    /// Test double-stream attention mask pattern:
    /// Joint sequence order: [txt, ref, output]
    /// - txt queries: attend to ALL (txt, ref, output)
    /// - ref queries: attend to txt + ref ONLY (blocked from output)
    /// - output queries: attend to ALL
    @Test func doubleStreamMaskPattern() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = Flux2Attention(dim: 128, numHeads: 4, headDim: 32)
        let textLen = 3
        let refLen = 2
        let outputLen = 4
        let totalSeq = textLen + refLen + outputLen  // 9

        let mask = attn.buildKVExtractionMask(
            textLen: textLen, refLen: refLen, outputLen: outputLen, totalSeq: totalSeq
        )

        #expect(mask.shape == [1, 1, totalSeq, totalSeq])

        eval(mask)
        let flat = mask.reshaped([totalSeq, totalSeq])

        func maskVal(_ q: Int, _ k: Int) -> Float {
            flat[q, k].item(Float.self)
        }

        // txt queries (rows 0-2) should attend to everything -> all 0.0
        for q in 0..<textLen {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0, "txt query \(q) should attend to key \(k)")
            }
        }

        // ref queries (rows 3-4) should attend to txt+ref (0-4) but NOT output (5-8)
        for q in textLen..<(textLen + refLen) {
            for k in 0..<(textLen + refLen) {
                #expect(maskVal(q, k) == 0.0, "ref query \(q) should attend to key \(k)")
            }
            for k in (textLen + refLen)..<totalSeq {
                #expect(maskVal(q, k) == -Float.infinity, "ref query \(q) should be blocked from output key \(k)")
            }
        }

        // output queries (rows 5-8) should attend to everything -> all 0.0
        for q in (textLen + refLen)..<totalSeq {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0, "output query \(q) should attend to key \(k)")
            }
        }
    }

    /// Test single-stream attention mask pattern.
    @Test func singleStreamMaskPattern() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = Flux2ParallelSelfAttention(dim: 128, numHeads: 4, headDim: 32)
        let textLen = 3
        let refLen = 2
        let outputLen = 4
        let totalSeq = textLen + refLen + outputLen

        let mask = attn.buildSingleStreamKVExtractionMask(
            textLen: textLen, refLen: refLen, outputLen: outputLen, totalSeq: totalSeq
        )

        #expect(mask.shape == [1, 1, totalSeq, totalSeq])

        eval(mask)
        let flat = mask.reshaped([totalSeq, totalSeq])

        func maskVal(_ q: Int, _ k: Int) -> Float {
            flat[q, k].item(Float.self)
        }

        for q in textLen..<(textLen + refLen) {
            for k in (textLen + refLen)..<totalSeq {
                #expect(maskVal(q, k) == -Float.infinity)
            }
        }

        for q in 0..<textLen {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0)
            }
        }
        for q in (textLen + refLen)..<totalSeq {
            for k in 0..<totalSeq {
                #expect(maskVal(q, k) == 0.0)
            }
        }
    }

    @Test func maskWithZeroReferenceTokens() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = Flux2Attention(dim: 128, numHeads: 4, headDim: 32)
        let mask = attn.buildKVExtractionMask(textLen: 5, refLen: 0, outputLen: 10, totalSeq: 15)

        eval(mask)
        let flat = mask.reshaped([15 * 15])

        let sum = MLX.abs(flat).sum()
        eval(sum)
        #expect(sum.item(Float.self) == 0.0, "No ref tokens -> no blocking")
    }
}

// MARK: - Flux2Attention KV Methods Tests

@Suite struct Flux2AttentionKVTests {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128
    let seqLenTxt = 8
    let seqLenRef = 6
    let seqLenImg = 10

    func makeAttention() -> Flux2Attention {
        Flux2Attention(dim: dim, numHeads: numHeads, headDim: headDim)
    }

    func makeRoPE(seqLen: Int) -> (cos: MLXArray, sin: MLXArray) {
        (cos: MLXRandom.normal([seqLen, headDim]),
         sin: MLXRandom.normal([seqLen, headDim]))
    }

    @Test func callWithKVExtractionOutputShapes() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = makeAttention()
        let imgPlusRef = MLXRandom.normal([batchSize, seqLenRef + seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenRef + seqLenImg)

        let (hsOut, ehsOut, cache) = attn.callWithKVExtraction(
            hiddenStates: imgPlusRef,
            encoderHiddenStates: txt,
            rotaryEmb: rope,
            referenceTokenCount: seqLenRef
        )
        eval(hsOut, ehsOut, cache.keys, cache.values)

        #expect(hsOut.shape == [batchSize, seqLenRef + seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])

        #expect(cache.keys.shape == [batchSize, numHeads, seqLenRef, headDim])
        #expect(cache.values.shape == [batchSize, numHeads, seqLenRef, headDim])
    }

    @Test func callWithKVCachedOutputShapes() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = makeAttention()
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenImg)

        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let (hsOut, ehsOut) = attn.callWithKVCached(
            hiddenStates: img,
            encoderHiddenStates: txt,
            rotaryEmb: rope,
            cachedKV: cachedKV
        )
        eval(hsOut, ehsOut)

        #expect(hsOut.shape == [batchSize, seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])
    }

    @Test func standardAndCachedConsistentShapes() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = makeAttention()
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let rope = makeRoPE(seqLen: seqLenTxt + seqLenImg)

        let (stdHs, stdEhs) = attn.callAsFunction(
            hiddenStates: img, encoderHiddenStates: txt, rotaryEmb: rope
        )

        let emptyCache = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, 0, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, 0, headDim])
        )
        let (cachedHs, cachedEhs) = attn.callWithKVCached(
            hiddenStates: img, encoderHiddenStates: txt,
            rotaryEmb: rope, cachedKV: emptyCache
        )
        eval(stdHs, stdEhs, cachedHs, cachedEhs)

        #expect(stdHs.shape == cachedHs.shape)
        #expect(stdEhs.shape == cachedEhs.shape)
    }
}

// MARK: - Flux2ParallelSelfAttention KV Methods Tests

@Suite struct Flux2ParallelAttentionKVTests {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128
    let textLen = 8
    let refLen = 6
    let imgLen = 10

    func makeAttention() -> Flux2ParallelSelfAttention {
        Flux2ParallelSelfAttention(dim: dim, numHeads: numHeads, headDim: headDim)
    }

    @Test func callWithKVExtractionOutputShapes() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = makeAttention()
        let totalSeq = textLen + refLen + imgLen
        let combined = MLXRandom.normal([batchSize, totalSeq, dim])
        let rope = (cos: MLXRandom.normal([totalSeq, headDim]),
                    sin: MLXRandom.normal([totalSeq, headDim]))

        let (output, cache) = attn.callWithKVExtraction(
            hiddenStates: combined,
            rotaryEmb: rope,
            textLen: textLen,
            referenceTokenCount: refLen
        )
        eval(output, cache.keys, cache.values)

        #expect(output.shape == [batchSize, totalSeq, dim])

        #expect(cache.keys.shape == [batchSize, numHeads, refLen, headDim])
        #expect(cache.values.shape == [batchSize, numHeads, refLen, headDim])
    }

    @Test func callWithKVCachedOutputShapes() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let attn = makeAttention()
        let seqLen = textLen + imgLen  // no ref tokens
        let combined = MLXRandom.normal([batchSize, seqLen, dim])
        let rope = (cos: MLXRandom.normal([seqLen, headDim]),
                    sin: MLXRandom.normal([seqLen, headDim]))

        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, refLen, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, refLen, headDim])
        )

        let output = attn.callWithKVCached(
            hiddenStates: combined,
            rotaryEmb: rope,
            cachedKV: cachedKV,
            textLen: textLen
        )
        eval(output)

        #expect(output.shape == [batchSize, seqLen, dim])
    }
}

// MARK: - Transformer Block KV Forwarding Tests

@Suite struct TransformerBlockKVTests {

    let batchSize = 1
    let numHeads = 4
    let headDim = 32
    let dim = 128
    let seqLenTxt = 8
    let seqLenRef = 4
    let seqLenImg = 10

    @Test func doubleStreamBlockKVExtraction() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let block = Flux2TransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let imgPlusRef = MLXRandom.normal([batchSize, seqLenRef + seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let totalRope = seqLenTxt + seqLenRef + seqLenImg
        let rope = (cos: MLXRandom.normal([totalRope, headDim]),
                    sin: MLXRandom.normal([totalRope, headDim]))

        let (ehsOut, hsOut, cache) = block.callWithKVExtraction(
            hiddenStates: imgPlusRef,
            encoderHiddenStates: txt,
            temb: temb,
            rotaryEmb: rope,
            imgModParams: nil,
            txtModParams: nil,
            referenceTokenCount: seqLenRef
        )
        eval(ehsOut, hsOut, cache.keys, cache.values)

        #expect(hsOut.shape == [batchSize, seqLenRef + seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])
        #expect(cache.keys.shape == [batchSize, numHeads, seqLenRef, headDim])
    }

    @Test func doubleStreamBlockKVCached() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let block = Flux2TransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let img = MLXRandom.normal([batchSize, seqLenImg, dim])
        let txt = MLXRandom.normal([batchSize, seqLenTxt, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([seqLenTxt + seqLenImg, headDim]),
                    sin: MLXRandom.normal([seqLenTxt + seqLenImg, headDim]))
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let (ehsOut, hsOut) = block.callWithKVCached(
            hiddenStates: img,
            encoderHiddenStates: txt,
            temb: temb,
            rotaryEmb: rope,
            imgModParams: nil,
            txtModParams: nil,
            cachedKV: cachedKV
        )
        eval(ehsOut, hsOut)

        #expect(hsOut.shape == [batchSize, seqLenImg, dim])
        #expect(ehsOut.shape == [batchSize, seqLenTxt, dim])
    }

    @Test func singleStreamBlockKVExtraction() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let block = Flux2SingleTransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let totalSeq = seqLenTxt + seqLenRef + seqLenImg
        let combined = MLXRandom.normal([batchSize, totalSeq, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([totalSeq, headDim]),
                    sin: MLXRandom.normal([totalSeq, headDim]))

        let (output, cache) = block.callWithKVExtraction(
            hiddenStates: combined,
            temb: temb,
            rotaryEmb: rope,
            modParams: nil,
            textLen: seqLenTxt,
            referenceTokenCount: seqLenRef
        )
        eval(output, cache.keys, cache.values)

        #expect(output.shape == [batchSize, totalSeq, dim])
        #expect(cache.keys.shape == [batchSize, numHeads, seqLenRef, headDim])
    }

    @Test func singleStreamBlockKVCached() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let block = Flux2SingleTransformerBlock(dim: dim, numHeads: numHeads, headDim: headDim)
        let seqLen = seqLenTxt + seqLenImg  // no ref
        let combined = MLXRandom.normal([batchSize, seqLen, dim])
        let temb = MLXRandom.normal([batchSize, dim])
        let rope = (cos: MLXRandom.normal([seqLen, headDim]),
                    sin: MLXRandom.normal([seqLen, headDim]))
        let cachedKV = LayerKVCacheEntry(
            keys: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim]),
            values: MLXRandom.normal([batchSize, numHeads, seqLenRef, headDim])
        )

        let output = block.callWithKVCached(
            hiddenStates: combined,
            temb: temb,
            rotaryEmb: rope,
            modParams: nil,
            cachedKV: cachedKV,
            textLen: seqLenTxt
        )
        eval(output)

        #expect(output.shape == [batchSize, seqLen, dim])
    }
}

// MARK: - SchedulerScaleNoise (requires eval to compare values)

@Suite struct SchedulerEvalTests {

    @Test func schedulerScaleNoise() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }
        let scheduler = FlowMatchEulerScheduler()

        let sample = MLXArray([Float(1.0)])
        let noise = MLXArray([Float(0.0)])

        // At sigma = 0, should return sample unchanged
        let result = scheduler.scaleNoise(sample: sample, sigma: 0.0, noise: noise)
        eval(result)
        #expect(abs(result.item(Float.self) - 1.0) < 0.001)
    }
}
