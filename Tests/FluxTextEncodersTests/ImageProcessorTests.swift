/**
 * ImageProcessorTests.swift
 * Unit tests for ImageProcessor and ImageProcessorConfig
 */

import CoreGraphics
import Foundation
import MLX
import Testing

@testable import FluxTextEncoders

@Suite("ImageProcessorTests")
struct ImageProcessorTests {

  // MARK: - ImageProcessorConfig Tests

  @Test func pixtralConfigDefaults() {
    let config = ImageProcessorConfig.pixtral

    #expect(config.imageSize == 1540, "Pixtral image size should be 1540")
    #expect(config.patchSize == 14, "Pixtral patch size should be 14")
    #expect(
      abs(config.rescaleFactor - (1.0 / 255.0)) < 0.0001,
      "Rescale factor should be 1/255")
  }

  @Test func pixtralConfigImageMean() {
    let config = ImageProcessorConfig.pixtral

    #expect(config.imageMean.count == 3, "Should have 3 mean values (RGB)")
    #expect(abs(config.imageMean[0] - 0.48145466) < 0.0001, "R mean")
    #expect(abs(config.imageMean[1] - 0.4578275) < 0.0001, "G mean")
    #expect(abs(config.imageMean[2] - 0.40821073) < 0.0001, "B mean")
  }

  @Test func pixtralConfigImageStd() {
    let config = ImageProcessorConfig.pixtral

    #expect(config.imageStd.count == 3, "Should have 3 std values (RGB)")
    #expect(abs(config.imageStd[0] - 0.26862954) < 0.0001, "R std")
    #expect(abs(config.imageStd[1] - 0.26130258) < 0.0001, "G std")
    #expect(abs(config.imageStd[2] - 0.27577711) < 0.0001, "B std")
  }

  @Test func customConfigInit() {
    let config = ImageProcessorConfig(
      imageSize: 224,
      patchSize: 16,
      imageMean: [0.5, 0.5, 0.5],
      imageStd: [0.5, 0.5, 0.5],
      rescaleFactor: 1.0 / 255.0
    )

    #expect(config.imageSize == 224)
    #expect(config.patchSize == 16)
    #expect(config.imageMean == [0.5, 0.5, 0.5])
    #expect(config.imageStd == [0.5, 0.5, 0.5])
  }

  // MARK: - ImageProcessor Tests

  @Test func imageProcessorInitWithDefaultConfig() {
    let processor = ImageProcessor()

    #expect(
      processor.config.imageSize == 1540,
      "Default config should be Pixtral")
  }

  @Test func imageProcessorInitWithCustomConfig() {
    let customConfig = ImageProcessorConfig(
      imageSize: 384,
      patchSize: 14,
      imageMean: [0.5, 0.5, 0.5],
      imageStd: [0.5, 0.5, 0.5],
      rescaleFactor: 1.0 / 255.0
    )
    let processor = ImageProcessor(config: customConfig)

    #expect(processor.config.imageSize == 384)
  }

  @Test func getNumPatches() {
    let processor = ImageProcessor()

    // Test with dimensions divisible by patch size
    let (patchesX, patchesY, total) = processor.getNumPatches(width: 224, height: 224)

    #expect(patchesX == 224 / 14, "Width patches")
    #expect(patchesY == 224 / 14, "Height patches")
    #expect(total == patchesX * patchesY, "Total patches")
  }

  @Test func getNumPatchesVariousSizes() {
    let processor = ImageProcessor()

    // 336x336
    let (px1, py1, t1) = processor.getNumPatches(width: 336, height: 336)
    #expect(px1 == 24)
    #expect(py1 == 24)
    #expect(t1 == 576)

    // 448x336 (rectangular)
    let (px2, py2, t2) = processor.getNumPatches(width: 448, height: 336)
    #expect(px2 == 32)
    #expect(py2 == 24)
    #expect(t2 == 768)
  }

  // MARK: - Error Tests

  @Test func imageProcessorErrorDescriptions() {
    #expect(ImageProcessorError.invalidImage.errorDescription == "Invalid image format")
    #expect(
      ImageProcessorError.contextCreationFailed.errorDescription
        == "Failed to create graphics context")
    #expect(ImageProcessorError.unsupportedFormat.errorDescription == "Unsupported image format")

    let fileNotFound = ImageProcessorError.fileNotFound("/path/to/image.jpg")
    #expect(fileNotFound.errorDescription == "Image file not found: /path/to/image.jpg")
  }

  @Test func loadImageFromNonExistentPath() {
    let processor = ImageProcessor()

    #expect(throws: (any Error).self) {
      try processor.loadImageAsCGImage(from: "/nonexistent/path/image.jpg")
    }
  }

  @Test func preprocessFromFileNonExistent() {
    let processor = ImageProcessor()

    #expect(throws: (any Error).self) { try processor.preprocessFromFile("/nonexistent.jpg") }
  }

  // MARK: - Patch Calculation Tests

  @Test func patchCalculationWithPixtralConfig() {
    let processor = ImageProcessor(config: .pixtral)

    // Maximum size (1540x1540)
    let (maxPX, maxPY, maxTotal) = processor.getNumPatches(width: 1540, height: 1540)
    #expect(maxPX == 110)  // 1540 / 14
    #expect(maxPY == 110)
    #expect(maxTotal == 12100)
  }

  @Test func patchCalculationMinimum() {
    let processor = ImageProcessor()

    // Single patch (14x14)
    let (px, py, total) = processor.getNumPatches(width: 14, height: 14)
    #expect(px == 1)
    #expect(py == 1)
    #expect(total == 1)
  }

  // MARK: - Config Codable Tests

  @Test func imageProcessorConfigCodable() throws {
    let config = ImageProcessorConfig.pixtral

    let encoder = JSONEncoder()
    let data = try encoder.encode(config)

    let decoder = JSONDecoder()
    let decoded = try decoder.decode(ImageProcessorConfig.self, from: data)

    #expect(decoded.imageSize == config.imageSize)
    #expect(decoded.patchSize == config.patchSize)
    #expect(decoded.imageMean == config.imageMean)
    #expect(decoded.imageStd == config.imageStd)
    #expect(decoded.rescaleFactor == config.rescaleFactor)
  }

}

// MARK: - preprocess(_:) on synthetic CGImages (CPU-side resize + normalize)

@Suite("ImageProcessorPreprocessTests")
struct ImageProcessorPreprocessTests {

  /// Solid-color, opaque DeviceRGB image.
  private func solidImage(width: Int, height: Int, rgb: (CGFloat, CGFloat, CGFloat)) throws
    -> CGImage
  {
    let context = try #require(
      CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.setFillColor(red: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return try #require(context.makeImage())
  }

  @Test(arguments: [
    // (source w, h, maxSize) -> (expected H, W)
    (100, 50, 1540, 56, 112),  // smaller than max: no upscale, pad up to /14
    (28, 28, 1540, 28, 28),  // already aligned
    (3000, 1500, 280, 140, 280),  // downscale longest edge to 280
    (1500, 3000, 280, 280, 140),  // portrait downscale
    (15, 1, 1540, 14, 28),  // tiny: each edge rounds up to one patch multiple
  ])
  func outputShapeIsNHWCAlignedToPatchSize(
    width: Int, height: Int, maxSize: Int, expectedH: Int, expectedW: Int
  ) throws {
    let processor = ImageProcessor()
    let image = try solidImage(width: width, height: height, rgb: (0.5, 0.5, 0.5))

    let out = try processor.preprocess(image, maxSize: maxSize)

    #expect(out.shape == [1, expectedH, expectedW, 3])
    #expect(out.shape[1] % processor.config.patchSize == 0)
    #expect(out.shape[2] % processor.config.patchSize == 0)
  }

  @Test func defaultPreprocessUsesConfigImageSize() throws {
    let config = ImageProcessorConfig(
      imageSize: 56, patchSize: 14, imageMean: [0, 0, 0], imageStd: [1, 1, 1],
      rescaleFactor: 1.0 / 255.0)
    let processor = ImageProcessor(config: config)
    let image = try solidImage(width: 200, height: 100, rgb: (0, 0, 0))

    let viaDefault = try processor.preprocess(image)
    let viaExplicit = try processor.preprocess(image, maxSize: 56)

    #expect(viaDefault.shape == [1, 28, 56, 3])
    #expect(viaDefault.shape == viaExplicit.shape)
  }

  @Test func identityNormalizationYieldsRescaledPixels() throws {
    // mean 0 / std 1 → output is just pixel/255, so white → 1 and black → 0.
    let config = ImageProcessorConfig(
      imageSize: 1540, patchSize: 14, imageMean: [0, 0, 0], imageStd: [1, 1, 1],
      rescaleFactor: 1.0 / 255.0)
    let processor = ImageProcessor(config: config)

    let white = try processor.preprocess(try solidImage(width: 28, height: 28, rgb: (1, 1, 1)))
    let black = try processor.preprocess(try solidImage(width: 28, height: 28, rgb: (0, 0, 0)))

    #expect(abs(white.min().item(Float.self) - 1) < 1e-4)
    #expect(abs(white.max().item(Float.self) - 1) < 1e-4)
    #expect(abs(black.max().item(Float.self)) < 1e-4)
  }

  @Test func pixtralNormalizationAppliesPerChannelMeanAndStd() throws {
    let processor = ImageProcessor()  // .pixtral
    let config = processor.config
    // Pure red: R=1, G=0, B=0 before normalization. 28×28 needs no resize.
    let out = try processor.preprocess(try solidImage(width: 28, height: 28, rgb: (1, 0, 0)))

    let pixel = out[0, 14, 14].asArray(Float.self)
    let expected = [
      (1 - config.imageMean[0]) / config.imageStd[0],
      (0 - config.imageMean[1]) / config.imageStd[1],
      (0 - config.imageMean[2]) / config.imageStd[2],
    ]
    for c in 0..<3 {
      #expect(abs(pixel[c] - expected[c]) < 1e-3, "channel \(c): \(pixel[c]) vs \(expected[c])")
    }
  }
}
