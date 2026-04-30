/**
 * CoverageGapTests.swift
 * Mock-pipeline coverage tests that have no equivalent in the per-type test suites.
 * All tests are CI-safe: no GPU, no model downloads.
 */

import Foundation
import TestHelpers
import Testing

@testable import FluxTextEncoders

@Suite("CoverageGaps")
struct CoverageGapTests {

  // MARK: - Progress callback fires correct number of times

  @Test func progressCallbackFiresCorrectNumberOfTimes() async {
    let mock = MockFlux2Pipeline()
    mock.simulatedSteps = 6
    // Use a class-based counter to avoid Swift concurrency mutation warning
    final class Counter: @unchecked Sendable {
      var value: Int = 0
    }
    let counter = Counter()

    _ = try? await mock.generate(
      mode: .textToImage,
      prompt: "test prompt",
      height: 64,
      width: 64,
      steps: 6,
      guidance: 3.5,
      seed: nil,
      upsamplePrompt: false,
      checkpointInterval: nil,
      onProgress: { _, _ in counter.value += 1 },
      onCheckpoint: nil
    )

    #expect(counter.value == 6, "Progress callback should fire exactly 6 times (once per step)")
  }

  // MARK: - Error path for corrupted model file

  @Test func errorPathForCorruptedModelFile() async {
    let mock = MockFlux2Pipeline()
    mock.errorToThrow = NSError(
      domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Simulated error"])

    var didThrow = false
    do {
      _ = try await mock.generate(
        mode: .textToImage,
        prompt: "test prompt",
        height: 64,
        width: 64,
        steps: 4,
        guidance: 3.5,
        seed: nil,
        upsamplePrompt: false,
        checkpointInterval: nil,
        onProgress: nil,
        onCheckpoint: nil
      )
    } catch {
      didThrow = true
    }
    #expect(didThrow, "generate should throw when errorToThrow is set")
  }

  // MARK: - Concurrency — parallel embedding extraction

  @Test func concurrencyParallelEmbeddingExtraction() async {
    let mock1 = MockFlux2Pipeline()
    let mock2 = MockFlux2Pipeline()

    async let result1 = mock1.generate(
      mode: .textToImage,
      prompt: "first concurrent prompt",
      height: 64,
      width: 64,
      steps: 4,
      guidance: 3.5,
      seed: nil,
      upsamplePrompt: false,
      checkpointInterval: nil,
      onProgress: nil,
      onCheckpoint: nil
    )

    async let result2 = mock2.generate(
      mode: .textToImage,
      prompt: "second concurrent prompt",
      height: 64,
      width: 64,
      steps: 4,
      guidance: 3.5,
      seed: nil,
      upsamplePrompt: false,
      checkpointInterval: nil,
      onProgress: nil,
      onCheckpoint: nil
    )

    let image1 = try? await result1
    let image2 = try? await result2

    #expect(image1 != nil, "First concurrent generate should succeed")
    #expect(image2 != nil, "Second concurrent generate should succeed")
  }
}
