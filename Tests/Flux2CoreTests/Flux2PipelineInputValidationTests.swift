// Flux2PipelineInputValidationTests.swift
//
// `Flux2Pipeline` input validation that runs BEFORE any weight load, so it is
// CI-safe (no GPU work, no model files). Each refusal must throw
// `Flux2Error.invalidConfiguration` and emit the matching `errorThrown`
// telemetry event (AGENTS.md §11: `errorThrown` precedes every `throw`).
//
// Deliberately NOT covered here: image-size / RAM-tier checks. Those are
// advisory by design (see `generateWithResult`) and must never refuse a
// generation, and proving "does not throw" would require loading weights.

import CoreGraphics
import Foundation
import TestHelpers
import Testing

@testable import Flux2Core

@Suite("Flux2Pipeline pre-load input validation")
struct Flux2PipelineInputValidationTests {

  private func isInvalidConfiguration(_ error: any Error, containing text: String) -> Bool {
    guard case Flux2Error.invalidConfiguration(let message) = error else { return false }
    return message.contains(text)
  }

  private func hasInvalidConfigurationEvent(
    _ events: [Flux2TelemetryEvent], containing text: String
  ) -> Bool {
    events.contains {
      if case .errorThrown(phase: .invalidConfiguration, let description) = $0 {
        return description.contains(text)
      }
      return false
    }
  }

  @Test(arguments: [0, 4])
  func imageToImageRejectsReferenceCountOutsideOneToThree(count: Int) async {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)
    let images = Array(repeating: TestImage.make(width: 16, height: 16), count: count)

    await #expect {
      _ = try await pipeline.generateImageToImage(prompt: "test", images: images)
    } throws: { error in
      self.isInvalidConfiguration(error, containing: "Provide 1-3 reference images")
    }
    // Direct await before the throw — already captured, no wait needed.
    let events = await reporter.snapshot()
    #expect(hasInvalidConfigurationEvent(events, containing: "Provide 1-3 reference images"))
    #expect(!pipeline.isLoaded, "validation must fail before any model load")
  }

  @Test func imageToImageWithResultRejectsTooManyReferences() async {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)
    let images = Array(repeating: TestImage.make(width: 16, height: 16), count: 4)

    await #expect {
      _ = try await pipeline.generateImageToImageWithResult(prompt: "test", images: images)
    } throws: { error in
      self.isInvalidConfiguration(error, containing: "Provide 1-3 reference images")
    }
    let events = await reporter.snapshot()
    #expect(hasInvalidConfigurationEvent(events, containing: "Provide 1-3 reference images"))
  }

  @Test func imageToImageRejectsUndecodableImageData() async {
    let reporter = MockFlux2TelemetryReporter()
    let pipeline = Flux2Pipeline(model: .klein4B, quantization: .minimal)
    pipeline.setTelemetry(reporter)
    let valid = TestImage.encode(TestImage.make(width: 16, height: 16))!
    let garbage = Data([0x00, 0x01, 0x02, 0x03])

    await #expect {
      _ = try await pipeline.generateImageToImage(prompt: "test", imageData: [valid, garbage])
    } throws: { error in
      self.isInvalidConfiguration(error, containing: "Failed to decode image data at index 1")
    }
    // The decode-failure emit is a fire-and-forget Task (the map closure is
    // synchronous), so await its delivery instead of snapshotting immediately.
    let sawEvent = await reporter.waitFor { events in
      events.contains {
        if case .errorThrown(phase: .invalidConfiguration, let description) = $0 {
          return description.contains("Failed to decode image data at index 1")
        }
        return false
      }
    }
    #expect(sawEvent)
    withExtendedLifetime(pipeline) {}
  }
}
