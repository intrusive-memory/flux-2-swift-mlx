/**
 * TextEncoderModelDirectoryTests.swift
 * Tests for custom model directory configuration in TextEncoderModelDownloader
 */

import Testing
import Foundation
@testable import FluxTextEncoders

@Suite("TextEncoderModelDirectoryTests", .serialized)
struct TextEncoderModelDirectoryTests {

    init() {
        // Reset to default before each test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    // MARK: - Default Directory

    @Test func defaultModelsDirectoryIsMistralModels() {
        #if os(macOS)
        let expected = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mistral")
            .appendingPathComponent("models")
        #else
        let expected = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("mistral")
            .appendingPathComponent("models")
        #endif
        #expect(TextEncoderModelDownloader.modelsDirectory == expected)
    }

    // MARK: - Custom Directory

    @Test func customModelsDirectoryOverridesDefault() {
        let custom = URL(fileURLWithPath: "/tmp/test-text-models")
        TextEncoderModelDownloader.customModelsDirectory = custom
        #expect(TextEncoderModelDownloader.modelsDirectory == custom)
        // Reset after test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    @Test func customModelsDirectoryNilFallsBackToDefault() {
        let custom = URL(fileURLWithPath: "/tmp/test-text-models")
        TextEncoderModelDownloader.customModelsDirectory = custom
        #expect(TextEncoderModelDownloader.modelsDirectory == custom)

        TextEncoderModelDownloader.customModelsDirectory = nil
        #if os(macOS)
        let expected = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mistral")
            .appendingPathComponent("models")
        #else
        let expected = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("mistral")
            .appendingPathComponent("models")
        #endif
        #expect(TextEncoderModelDownloader.modelsDirectory == expected)
    }

    // MARK: - reconfigureHubApi

    @Test func reconfigureHubApiDoesNotCrash() {
        // Verify reconfigureHubApi can be called without errors
        TextEncoderModelDownloader.customModelsDirectory = URL(fileURLWithPath: "/tmp/test-models")
        TextEncoderModelDownloader.reconfigureHubApi()

        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    // MARK: - hubCachePath with custom directory

    @Test func hubCachePathUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        // Create a fake model directory for a known repo
        let model = ModelInfo(
            id: "test",
            repoId: "test-org/test-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("test-model")

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        let cachePath = TextEncoderModelDownloader.hubCachePath(for: model)
        #expect(cachePath != nil)
        #expect(cachePath!.standardizedFileURL.path == modelDir.standardizedFileURL.path)

        // Reset after test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    @Test func hubCachePathReturnsNilWhenModelNotInCustomDir() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir

        let model = ModelInfo(
            id: "test",
            repoId: "test-org/missing-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        let cachePath = TextEncoderModelDownloader.hubCachePath(for: model)
        #expect(cachePath == nil)

        // Reset after test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    // MARK: - findModelPath with custom directory

    @Test func findModelPathUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        let model = ModelInfo(
            id: "test",
            repoId: "test-org/test-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        // Create fake model in hub download location (custom dir)
        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("test-model")

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir
        TextEncoderModelDownloader.reconfigureHubApi()

        let found = TextEncoderModelDownloader.findModelPath(for: model)
        #expect(found != nil)
        #expect(found!.path.hasPrefix(tempDir.path))

        // Reset after test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    // MARK: - findQwen3ModelPath with custom directory

    @Test func findQwen3ModelPathUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        let model = Qwen3ModelInfo(
            id: "test-qwen",
            repoId: "test-org/qwen3-test",
            name: "Qwen3 Test",
            description: "Test Qwen3 model",
            variant: .qwen3_4B_8bit,
            parameters: "4B"
        )

        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("qwen3-test")

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir
        TextEncoderModelDownloader.reconfigureHubApi()

        let found = TextEncoderModelDownloader.findQwen3ModelPath(for: model)
        #expect(found != nil)
        #expect(found!.path.hasPrefix(tempDir.path))

        // Reset after test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }

    // MARK: - Multiple switches

    @Test func switchingCustomDirectories() {
        let dir1 = URL(fileURLWithPath: "/tmp/models-a")
        let dir2 = URL(fileURLWithPath: "/tmp/models-b")

        TextEncoderModelDownloader.customModelsDirectory = dir1
        #expect(TextEncoderModelDownloader.modelsDirectory == dir1)

        TextEncoderModelDownloader.customModelsDirectory = dir2
        #expect(TextEncoderModelDownloader.modelsDirectory == dir2)

        TextEncoderModelDownloader.customModelsDirectory = nil
        #if os(macOS)
        let defaultDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mistral")
            .appendingPathComponent("models")
        #else
        let defaultDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("mistral")
            .appendingPathComponent("models")
        #endif
        #expect(TextEncoderModelDownloader.modelsDirectory == defaultDir)
    }

    // MARK: - isModelDownloaded with custom directory

    @Test func isModelDownloadedUsesCustomDirectory() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux2-te-test-\(UUID().uuidString)")
            .appendingPathComponent("models")

        let model = ModelInfo(
            id: "test",
            repoId: "test-org/test-model",
            name: "Test",
            description: "Test model",
            variant: .mlx8bit,
            parameters: "1B"
        )

        // Empty custom dir — model should not be found
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir.deletingLastPathComponent())
        }

        TextEncoderModelDownloader.customModelsDirectory = tempDir
        TextEncoderModelDownloader.reconfigureHubApi()
        #expect(!TextEncoderModelDownloader.isModelDownloaded(model))

        // Add model files — now it should be found
        let modelDir = tempDir
            .appendingPathComponent("test-org")
            .appendingPathComponent("test-model")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try "{}".write(to: modelDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))

        TextEncoderModelDownloader.reconfigureHubApi()
        #expect(TextEncoderModelDownloader.isModelDownloaded(model))

        // Reset after test
        TextEncoderModelDownloader.customModelsDirectory = nil
        TextEncoderModelDownloader.reconfigureHubApi()
    }
}
