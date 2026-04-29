// ImageToImageGPUTests.swift — MLX-runtime tests for I2I latent helpers.
//
// Moved from Tests/Flux2CoreTests/ImageToImageTrainingTests.swift; these suites had
// `if isCI { return }` guards that bailed after allocating MLX arrays. Each test
// requires Metal eval to inspect array values, so they belong here.

import Testing
import Foundation
@testable import Flux2Core
import MLX

// MARK: - CachedLatentEntry I2I Tests

@Suite struct CachedLatentEntryI2ITests {

    @Test func cachedLatentEntryDefaultNoControl() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let entry = CachedLatentEntry(
            filename: "test.png",
            latent: MLXArray.zeros([1, 32, 64, 64]),
            width: 512,
            height: 512
        )

        #expect(entry.controlLatent == nil)
        #expect(entry.filename == "test.png")
        #expect(entry.width == 512)
        #expect(entry.height == 512)
    }

    @Test func cachedLatentEntryWithControlLatent() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let targetLatent = MLXArray.zeros([1, 32, 64, 64])
        let controlLatent = MLXArray.ones([1, 32, 64, 64])

        let entry = CachedLatentEntry(
            filename: "edit_001.png",
            latent: targetLatent,
            width: 512,
            height: 512,
            controlLatent: controlLatent
        )

        #expect(entry.controlLatent != nil)
        #expect(entry.controlLatent!.shape == [1, 32, 64, 64])
    }

    @Test func cachedLatentEntryControlLatentMatchesTargetShape() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let entry = CachedLatentEntry(
            filename: "pair.png",
            latent: target,
            width: w * 8,
            height: h * 8,
            controlLatent: control
        )

        #expect(entry.latent.shape == entry.controlLatent!.shape)
    }
}

// MARK: - Position ID Tests for I2I

@Suite struct PositionIDI2ITests {

    @Test func referenceImagePositionIDsGeneration() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let height = 512
        let width = 512
        let latentH = height / 8  // 64
        let latentW = width / 8   // 64

        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH,
            latentWidth: latentW,
            imageIndex: 0
        )

        let expectedPatches = latentH * latentW
        #expect(refImgIds.shape[0] == expectedPatches)
        #expect(refImgIds.shape[1] == 4)  // [T, H, W, L]
    }

    @Test func referenceImagePositionIDsTCoordinate() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let latentH = 64
        let latentW = 64

        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH,
            latentWidth: latentW,
            imageIndex: 0
        )

        eval(refImgIds)

        // T coordinate for first reference image should be 10 (T=10 + imageIndex*10)
        let firstT = refImgIds[0, 0].item(Int32.self)
        #expect(firstT == 10, "First reference image should have T=10")
    }

    @Test func referenceImagePositionIDsSecondImage() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let latentH = 64
        let latentW = 64

        let refIds0 = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH, latentWidth: latentW, imageIndex: 0
        )
        let refIds1 = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: latentH, latentWidth: latentW, imageIndex: 1
        )

        eval(refIds0, refIds1)

        let t0 = refIds0[0, 0].item(Int32.self)
        let t1 = refIds1[0, 0].item(Int32.self)

        #expect(t0 != t1)
    }

    @Test func concatenatedPositionIDs() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let height = 512
        let width = 512

        let imgIds = LatentUtils.generateImagePositionIDs(height: height, width: width)
        let refImgIds = LatentUtils.generateSingleReferenceImagePositionIDs(
            latentHeight: height / 8,
            latentWidth: width / 8,
            imageIndex: 0
        )

        let combined = concatenated([imgIds, refImgIds], axis: 0)

        let expectedTotal = imgIds.shape[0] + refImgIds.shape[0]
        #expect(combined.shape[0] == expectedTotal)
        #expect(combined.shape[1] == 4)
    }
}

// MARK: - Latent Packing I2I Tests

@Suite struct LatentPackingI2ITests {

    @Test func packedControlLatentSameShapeAsTarget() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let packedTarget = LatentUtils.packLatents(target, patchSize: 2)
        let packedControl = LatentUtils.packLatents(control, patchSize: 2)

        #expect(packedTarget.shape == packedControl.shape)
    }

    @Test func concatenatedLatentSequence() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let h = 64, w = 64
        let target = MLXRandom.normal([1, 32, h, w])
        let control = MLXRandom.normal([1, 32, h, w])

        let packedTarget = LatentUtils.packLatents(target, patchSize: 2)
        let packedControl = LatentUtils.packLatents(control, patchSize: 2)

        let combined = concatenated([packedTarget, packedControl], axis: 1)

        let outputSeqLen = packedTarget.shape[1]
        let totalSeqLen = combined.shape[1]

        #expect(totalSeqLen == 2 * outputSeqLen)
        #expect(combined.shape[0] == 1)  // batch
        #expect(combined.shape[2] == 128)  // channels
    }

    @Test func outputSlicingFromCombinedLatent() {
        guard checkGPUPreconditions(minimumBytes: 0) else { return }

        let seqLen = 1024
        let channels = 128

        let modelOutput = MLXRandom.normal([1, 2 * seqLen, channels])
        let outputPortion = modelOutput[0..., 0..<seqLen, 0...]

        eval(outputPortion)
        #expect(outputPortion.shape == [1, seqLen, channels])
    }
}
