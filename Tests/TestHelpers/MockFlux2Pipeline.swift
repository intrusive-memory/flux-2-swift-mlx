// MockFlux2Pipeline.swift - Lightweight mock for Flux2Pipeline.generate
// Framework-agnostic mock. Intended to be consumed by test targets.

import CoreGraphics
import Flux2Core

/// A mock that mirrors `Flux2Pipeline.generate` for unit tests.
///
/// Configure `simulatedSteps` and `errorToThrow` before calling `generate`.
/// The mock never loads models or touches the GPU.
public final class MockFlux2Pipeline: @unchecked Sendable {

    /// Number of fake denoising steps reported through `onProgress`.
    public var simulatedSteps: Int = 4

    /// If non-nil, `generate` throws this instead of returning an image.
    public var errorToThrow: (any Error)?

    /// Number of times `generate` has been called (useful for assertions).
    public private(set) var generateCallCount: Int = 0

    public init() {}

    // MARK: - generate (exact parameter list copied from Flux2Pipeline)

    /// Signature matches `Flux2Pipeline.generate` exactly.
    public func generate(
        mode: Flux2GenerationMode,
        prompt: String,
        interpretImagePaths: [String]? = nil,
        height: Int,
        width: Int,
        steps: Int,
        guidance: Float,
        seed: UInt64?,
        upsamplePrompt: Bool,
        checkpointInterval: Int?,
        onProgress: Flux2ProgressCallback?,
        onCheckpoint: Flux2CheckpointCallback?
    ) async throws -> CGImage {
        generateCallCount += 1

        if let error = errorToThrow {
            throw error
        }

        for i in 1...simulatedSteps {
            onProgress?(i, simulatedSteps)
        }

        return TestImage.make(width: width, height: height)
    }
}
