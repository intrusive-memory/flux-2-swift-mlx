// TrainingDatasetTests.swift - Dataset loading, batching and step accounting.
//
// Uses a temp-dir fixture of tiny solid-color PNGs + .txt captions, so image
// decode, center-crop/resize, [0,1] normalization and batch stacking run for
// real on the CPU-side path without any model weights.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing
import UniformTypeIdentifiers

@testable import Flux2Core

@Suite final class TrainingDatasetTests {

  let tempDir: URL
  let outputDir: URL

  init() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("TrainingDatasetTests_\(UUID().uuidString)")
    tempDir = root.appendingPathComponent("dataset")
    outputDir = root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  deinit {
    try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
  }

  // MARK: - Fixture helpers

  private func writePNG(
    _ name: String, width: Int = 48, height: Int = 48,
    rgb: (CGFloat, CGFloat, CGFloat) = (1, 0, 0)
  ) throws {
    let context = try #require(
      CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.setFillColor(red: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let image = try #require(context.makeImage())

    let url = tempDir.appendingPathComponent(name) as CFURL
    let dest = try #require(
      CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(dest, image, nil)
    #expect(CGImageDestinationFinalize(dest))
  }

  private func writeCaption(_ text: String, for imageName: String) throws {
    let base = (imageName as NSString).deletingPathExtension
    try text.write(
      to: tempDir.appendingPathComponent("\(base).txt"), atomically: true, encoding: .utf8)
  }

  private func addSample(
    _ name: String, caption: String, width: Int = 48, height: Int = 48,
    rgb: (CGFloat, CGFloat, CGFloat) = (1, 0, 0)
  ) throws {
    try writePNG(name, width: width, height: height, rgb: rgb)
    try writeCaption(caption, for: name)
  }

  private func makeConfig(
    triggerWord: String? = nil,
    imageSize: Int = 32,
    enableBucketing: Bool = false,
    bucketResolutions: [Int] = [512],
    batchSize: Int = 1,
    epochs: Int = 10,
    maxSteps: Int? = nil
  ) -> LoRATrainingConfig {
    LoRATrainingConfig(
      datasetPath: tempDir,
      captionExtension: "txt",
      triggerWord: triggerWord,
      imageSize: imageSize,
      enableBucketing: enableBucketing,
      bucketResolutions: bucketResolutions,
      shuffleDataset: false,
      batchSize: batchSize,
      epochs: epochs,
      maxSteps: maxSteps,
      outputPath: outputDir
    )
  }

  // MARK: - Construction

  @Test func emptyDatasetThrows() {
    #expect {
      _ = try TrainingDataset(config: self.makeConfig())
    } throws: { error in
      guard case TrainingDatasetError.emptyDataset = error else { return false }
      return true
    }
  }

  @Test func loadsCaptionsWithTriggerWord() throws {
    try addSample("a.png", caption: "photo of [trigger]")
    try addSample("b.png", caption: "another")

    let dataset = try TrainingDataset(config: makeConfig(triggerWord: "sks"))

    #expect(dataset.count == 2)
    #expect(Set(dataset.allCaptions) == ["photo of sks", "another"])
    #expect(Set(dataset.sampleMetadata.map(\.filename)) == ["a.png", "b.png"])
    #expect(dataset.buckets.isEmpty)  // bucketing disabled
  }

  // MARK: - Step accounting

  @Test(arguments: [
    (5, 2, 4, nil as Int?, 3, 12),  // ceil(5/2)=3 batches × 4 epochs
    (4, 2, 3, nil, 2, 6),
    (5, 1, 2, nil, 5, 10),
    (5, 2, 4, 7, 3, 7),  // maxSteps overrides epochs
  ])
  func batchesPerEpochAndTotalSteps(
    samples: Int, batchSize: Int, epochs: Int, maxSteps: Int?,
    expectedBatches: Int, expectedSteps: Int
  ) throws {
    for i in 0..<samples {
      try addSample("img\(i).png", caption: "c\(i)", width: 8, height: 8)
    }
    let dataset = try TrainingDataset(
      config: makeConfig(batchSize: batchSize, epochs: epochs, maxSteps: maxSteps))

    #expect(dataset.batchesPerEpoch == expectedBatches)
    #expect(dataset.totalSteps == expectedSteps)
  }

  // MARK: - Unbucketed batching

  @Test func unbucketedBatchesCoverEverySampleOnceThenEnd() throws {
    for i in 0..<3 {
      try addSample("img\(i).png", caption: "caption \(i)")
    }
    let dataset = try TrainingDataset(config: makeConfig(imageSize: 32, batchSize: 2))

    let first = try #require(try dataset.nextBatch())
    #expect(first.count == 2)
    #expect(first.images.shape == [2, 32, 32, 3])
    #expect(first.resolution == nil)
    #expect(first.width == 32 && first.height == 32)

    let second = try #require(try dataset.nextBatch())
    #expect(second.count == 1)
    #expect(second.images.shape == [1, 32, 32, 3])

    #expect(try dataset.nextBatch() == nil)

    #expect(
      Set(first.filenames + second.filenames) == ["img0.png", "img1.png", "img2.png"])
    // Captions travel with their filenames.
    for (name, caption) in zip(first.filenames + second.filenames, first.captions + second.captions)
    {
      let index = name.dropFirst(3).prefix(1)
      #expect(caption == "caption \(index)")
    }
  }

  @Test func startEpochRestartsIteration() throws {
    try addSample("a.png", caption: "a")
    let dataset = try TrainingDataset(config: makeConfig())

    #expect(dataset.currentEpoch == 0)
    #expect(try dataset.nextBatch() != nil)
    #expect(try dataset.nextBatch() == nil)

    dataset.startEpoch()

    #expect(dataset.currentEpoch == 1)
    #expect(try dataset.nextBatch() != nil)
  }

  @Test func imagePixelsAreNormalizedToUnitRange() throws {
    try addSample("red.png", caption: "red", rgb: (1, 0, 0))
    try addSample("gray.png", caption: "gray", rgb: (0.5, 0.5, 0.5))
    let dataset = try TrainingDataset(config: makeConfig(imageSize: 16))

    for index in 0..<dataset.count {
      let sample = try dataset.getSample(at: index)
      #expect(sample.image.shape == [16, 16, 3])
      #expect(sample.originalSize.width == 16 && sample.originalSize.height == 16)

      let center = sample.image[8, 8].asArray(Float.self)
      #expect(sample.image.min().item(Float.self) >= 0)
      #expect(sample.image.max().item(Float.self) <= 1)
      if sample.filename == "red.png" {
        #expect(abs(center[0] - 1) < 0.03)
        #expect(center[1] < 0.03 && center[2] < 0.03)
      } else {
        for channel in center {
          #expect(abs(channel - 0.5) < 0.03)
        }
      }
    }
  }

  @Test func nonSquareSourceIsCenterCroppedToTargetSize() throws {
    try addSample("wide.png", caption: "wide", width: 96, height: 32)
    let dataset = try TrainingDataset(config: makeConfig(imageSize: 24))

    let batch = try #require(try dataset.nextBatch())
    #expect(batch.images.shape == [1, 24, 24, 3])
  }

  @Test func getSampleOutOfBoundsThrows() throws {
    try addSample("a.png", caption: "a")
    let dataset = try TrainingDataset(config: makeConfig())

    for bad in [-1, 1] {
      #expect {
        _ = try dataset.getSample(at: bad)
      } throws: { error in
        guard case TrainingDatasetError.indexOutOfBounds(let i) = error else { return false }
        return i == bad
      }
    }
  }

  @Test func unreadableImageThrowsFailedToLoadImage() throws {
    try "not a png".write(
      to: tempDir.appendingPathComponent("broken.png"), atomically: true, encoding: .utf8)
    try writeCaption("broken", for: "broken.png")
    let dataset = try TrainingDataset(config: makeConfig())

    #expect {
      _ = try dataset.nextBatch()
    } throws: { error in
      guard case TrainingDatasetError.failedToLoadImage(let url) = error else { return false }
      return url.lastPathComponent == "broken.png"
    }
  }

  @Test func sequenceIterationYieldsEverySample() throws {
    try addSample("a.png", caption: "a")
    try addSample("b.png", caption: "b")
    let dataset = try TrainingDataset(config: makeConfig(imageSize: 8))

    let names = dataset.map(\.filename)
    #expect(Set(names) == ["a.png", "b.png"])
  }

  // MARK: - Target dimensions / bucketing

  @Test func targetDimensionsWithoutBucketingUseImageSize() throws {
    try addSample("wide.png", caption: "w", width: 96, height: 32)
    let dataset = try TrainingDataset(config: makeConfig(imageSize: 64))

    for name in ["wide.png", "unknown.png"] {
      let dims = dataset.getTargetDimensions(for: name)
      #expect(dims.width == 64 && dims.height == 64, "\(name)")
    }
  }

  @Test func bucketingAssignsByAspectRatioAndBatchesPerBucket() throws {
    try addSample("wide.png", caption: "w", width: 128, height: 64)
    try addSample("tall.png", caption: "t", width: 64, height: 128)
    try addSample("square.png", caption: "s", width: 64, height: 64)
    let dataset = try TrainingDataset(
      config: makeConfig(imageSize: 512, enableBucketing: true, bucketResolutions: [256]))

    let wide = dataset.getTargetDimensions(for: "wide.png")
    let tall = dataset.getTargetDimensions(for: "tall.png")
    let square = dataset.getTargetDimensions(for: "square.png")
    #expect(wide.width > wide.height)
    #expect(tall.height > tall.width)
    #expect(square.width == 256 && square.height == 256)
    // Unassigned filenames fall back to imageSize.
    let fallback = dataset.getTargetDimensions(for: "unknown.png")
    #expect(fallback.width == 512 && fallback.height == 512)

    #expect(dataset.buckets.count == 3)

    // One batch per bucket (batchSize 1, three distinct buckets), each at its
    // bucket's resolution.
    var seen: [String] = []
    while let batch = try dataset.nextBatch() {
      let bucket = try #require(batch.resolution)
      #expect(batch.images.shape == [1, bucket.height, bucket.width, 3])
      #expect(batch.width == bucket.width && batch.height == bucket.height)
      seen += batch.filenames
    }
    #expect(Set(seen) == ["wide.png", "tall.png", "square.png"])
  }

  // MARK: - Validation / statistics

  @Test func validateDelegatesToCaptionParser() throws {
    try addSample("a.png", caption: "a")
    try writePNG("uncaptioned.png")
    let dataset = try TrainingDataset(config: makeConfig())

    let result = dataset.validate()
    #expect(result.imageCount == 2)
    #expect(result.warnings.contains("Missing caption: uncaptioned.png"))
  }

  @Test func statisticsSummarizeCaptionLengths() throws {
    try addSample("a.png", caption: "ab")
    try addSample("b.png", caption: "abcd")
    try addSample("c.png", caption: "abcdefghi")
    let dataset = try TrainingDataset(config: makeConfig())

    let stats = dataset.getStatistics()
    #expect(stats.totalSamples == 3)
    #expect(stats.minCaptionLength == 2)
    #expect(stats.maxCaptionLength == 9)
    #expect(stats.avgCaptionLength == 5)  // (2+4+9)/3 = 5 (integer division)
    #expect(stats.summary.contains("min=2, max=9, avg=5"))
  }
}
