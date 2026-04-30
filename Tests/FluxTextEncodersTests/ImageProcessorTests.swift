/**
 * ImageProcessorTests.swift
 * Unit tests for ImageProcessor and ImageProcessorConfig
 */

import Foundation
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
      try processor.loadImage(from: "/nonexistent/path/image.jpg")
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
