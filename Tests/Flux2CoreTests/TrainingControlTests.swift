// TrainingControlTests.swift - Tests for training control functionality
// Copyright 2025 Vincent Gourbin

import Foundation
import Testing

@testable import Flux2Core

@Suite final class TrainingControlTests {

  var tempDir: URL

  init() throws {
    // Create a temporary directory for each test
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("TrainingControlTests_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  deinit {
    // Clean up temporary directory
    try? FileManager.default.removeItem(at: tempDir)
  }

  // MARK: - TrainingState Tests

  @Test func trainingStateInitialization() {
    let state = TrainingState(
      currentStep: 0,
      totalSteps: 1000,
      rngSeed: 42,
      configHash: "test_hash",
      modelType: "klein-4b",
      loraRank: 32,
      loraAlpha: 32.0
    )

    #expect(state.currentStep == 0)
    #expect(state.totalSteps == 1000)
    #expect(state.rngSeed == 42)
    #expect(state.modelType == "klein-4b")
    #expect(state.loraRank == 32)
    #expect(state.loraAlpha == 32.0)
    #expect(state.recentLosses.isEmpty)
    #expect(state.bestLoss == Float.infinity)
  }

  @Test func trainingStateRecordLoss() {
    var state = TrainingState(
      currentStep: 1,
      totalSteps: 100,
      rngSeed: 42,
      configHash: "test",
      modelType: "klein-4b",
      loraRank: 32,
      loraAlpha: 32.0
    )

    state.recordLoss(1.5)
    state.recordLoss(1.3)
    state.recordLoss(1.1)

    #expect(state.recentLosses.count == 3)
    #expect(state.bestLoss == 1.1)
    #expect(abs(state.averageLoss - (1.5 + 1.3 + 1.1) / 3.0) < 0.001)
  }

  @Test func trainingStateSaveAndLoad() throws {
    var state = TrainingState(
      currentStep: 50,
      totalSteps: 500,
      rngSeed: 123,
      configHash: "abc123",
      modelType: "klein-9b",
      loraRank: 16,
      loraAlpha: 16.0
    )
    state.recordLoss(1.2)
    state.recordLoss(1.0)
    state.recordCheckpoint(step: 50)

    // Save
    let saveURL = tempDir.appendingPathComponent("training_state.json")
    try state.save(to: saveURL)

    // Verify file exists
    #expect(FileManager.default.fileExists(atPath: saveURL.path))

    // Load
    let loadedState = try TrainingState.load(from: saveURL)

    #expect(loadedState.currentStep == 50)
    #expect(loadedState.totalSteps == 500)
    #expect(loadedState.rngSeed == 123)
    #expect(loadedState.modelType == "klein-9b")
    #expect(loadedState.loraRank == 16)
    #expect(loadedState.recentLosses.count == 2)
    #expect(loadedState.checkpointSteps == [50])
  }

  @Test func findLatestCheckpoint() throws {
    // Create checkpoint directories
    let checkpoint100 = tempDir.appendingPathComponent("checkpoint_000100")
    let checkpoint200 = tempDir.appendingPathComponent("checkpoint_000200")
    let checkpoint150 = tempDir.appendingPathComponent("checkpoint_000150")

    try FileManager.default.createDirectory(at: checkpoint100, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: checkpoint200, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: checkpoint150, withIntermediateDirectories: true)

    // Create training_state.json in each
    for (dir, step) in [(checkpoint100, 100), (checkpoint200, 200), (checkpoint150, 150)] {
      var state = TrainingState(
        currentStep: step,
        totalSteps: 500,
        rngSeed: 42,
        configHash: "test",
        modelType: "klein-4b",
        loraRank: 32,
        loraAlpha: 32.0
      )
      // Record a loss to avoid Float.infinity (which can't be serialized to JSON)
      state.recordLoss(1.0)
      try state.save(to: dir.appendingPathComponent("training_state.json"))
    }

    // Find latest
    let latest = TrainingState.findLatestCheckpoint(in: tempDir)

    #expect(latest != nil)
    #expect(latest?.step == 200)
  }

  @Test func findLatestCheckpointNoCheckpoints() {
    let latest = TrainingState.findLatestCheckpoint(in: tempDir)
    #expect(latest == nil)
  }

  @Test func configHash() {
    let hash1 = TrainingState.hashConfig(
      modelType: "klein-4b",
      rank: 32,
      alpha: 32.0,
      learningRate: 1e-4,
      datasetPath: "/path/to/dataset"
    )

    let hash2 = TrainingState.hashConfig(
      modelType: "klein-4b",
      rank: 32,
      alpha: 32.0,
      learningRate: 1e-4,
      datasetPath: "/path/to/dataset"
    )

    let hash3 = TrainingState.hashConfig(
      modelType: "klein-9b",  // Different model
      rank: 32,
      alpha: 32.0,
      learningRate: 1e-4,
      datasetPath: "/path/to/dataset"
    )

    #expect(hash1 == hash2)  // Same config = same hash
    #expect(hash1 != hash3)  // Different config = different hash
  }

  // MARK: - TrainingController Tests

  @Test func trainingControllerInitialization() {
    let controller = TrainingController(outputDirectory: tempDir)

    #expect(controller.status == .idle)
    #expect(controller.state == nil)
    #expect(controller.outputDirectory == tempDir)
  }

  @Test func trainingControllerPauseResume() {
    let controller = TrainingController(outputDirectory: tempDir)

    // Request pause
    controller.requestPause()

    // Check pause file exists
    let pauseFile = tempDir.appendingPathComponent(".pause")
    #expect(FileManager.default.fileExists(atPath: pauseFile.path))
    #expect(controller.shouldPause())

    // Resume
    controller.resume()

    // Check pause file removed
    #expect(!FileManager.default.fileExists(atPath: pauseFile.path))
    #expect(!controller.shouldPause())
  }

  @Test func trainingControllerStop() {
    let controller = TrainingController(outputDirectory: tempDir)

    // Request stop
    controller.requestStop()

    // Check stop file exists
    let stopFile = tempDir.appendingPathComponent(".stop")
    #expect(FileManager.default.fileExists(atPath: stopFile.path))
    #expect(controller.shouldStop())
  }

  @Test func trainingControllerForceStop() {
    let controller = TrainingController(outputDirectory: tempDir)

    #expect(!controller.shouldForceStop())

    controller.forceStop()

    #expect(controller.shouldForceStop())
    #expect(controller.shouldStop())  // Force stop also sets stop flag
  }

  @Test func trainingControllerCheckpointRequest() {
    let controller = TrainingController(outputDirectory: tempDir)

    #expect(!controller.shouldCheckpoint())

    controller.requestCheckpoint()

    #expect(controller.shouldCheckpoint())
    // Second call should return false (flag is cleared)
    #expect(!controller.shouldCheckpoint())
  }

  @Test func trainingControllerStateUpdate() {
    let controller = TrainingController(outputDirectory: tempDir)

    let state = TrainingState(
      currentStep: 100,
      totalSteps: 500,
      rngSeed: 42,
      configHash: "test",
      modelType: "klein-4b",
      loraRank: 32,
      loraAlpha: 32.0
    )

    controller.updateState(state)

    #expect(controller.state != nil)
    #expect(controller.state?.currentStep == 100)
  }

  @Test func trainingControllerStatusChange() {
    let controller = TrainingController(outputDirectory: tempDir)

    #expect(controller.status == .idle)

    controller.setStatus(.running)
    #expect(controller.status == .running)

    controller.setStatus(.paused)
    #expect(controller.status == .paused)

    controller.setStatus(.completed)
    #expect(controller.status == .completed)
  }

  @Test func trainingControllerStaticPauseResume() throws {
    // Test static CLI helper methods
    try TrainingController.pauseTraining(outputDir: tempDir)

    #expect(TrainingController.isPaused(outputDir: tempDir))

    try TrainingController.resumeTraining(outputDir: tempDir)

    #expect(!TrainingController.isPaused(outputDir: tempDir))
  }

  @Test func trainingControllerStaticStop() throws {
    try TrainingController.stopTraining(outputDir: tempDir)

    let stopFile = tempDir.appendingPathComponent(".stop")
    #expect(FileManager.default.fileExists(atPath: stopFile.path))
  }

  // MARK: - Observer Tests

  @Test func trainingObserver() {
    let controller = TrainingController(outputDirectory: tempDir)
    let observer = MockTrainingObserver()

    controller.addObserver(observer)

    // Trigger events
    controller.notifyStepCompleted(step: 10, totalSteps: 100, loss: 1.5)
    controller.setStatus(.paused)

    #expect(observer.lastStep == 10)
    #expect(observer.lastLoss == 1.5)
    #expect(observer.lastStatus == .paused)

    // Remove observer
    controller.removeObserver(observer)
    controller.notifyStepCompleted(step: 20, totalSteps: 100, loss: 1.0)

    // Should not be updated
    #expect(observer.lastStep == 10)
  }

  // MARK: - Pause Checkpoint Marker Tests

  @Test func pauseCheckpointMarker() throws {
    // Create a checkpoint directory with pause marker
    let checkpointDir = tempDir.appendingPathComponent("checkpoint_000100")
    try FileManager.default.createDirectory(at: checkpointDir, withIntermediateDirectories: true)

    let pauseMarker = checkpointDir.appendingPathComponent(".pause_checkpoint")
    FileManager.default.createFile(atPath: pauseMarker.path, contents: nil)

    // Verify marker exists
    #expect(FileManager.default.fileExists(atPath: pauseMarker.path))

    // Simulate cleanup after resume
    try FileManager.default.removeItem(at: checkpointDir)

    // Verify directory is removed
    #expect(!FileManager.default.fileExists(atPath: checkpointDir.path))
  }
}

// MARK: - Mock Observer

private class MockTrainingObserver: TrainingObserver {
  var lastStep: Int = 0
  var lastLoss: Float = 0
  var lastStatus: TrainingStatus = .idle
  var checkpointPaths: [URL] = []

  func trainingStatusChanged(_ status: TrainingStatus) {
    lastStatus = status
  }

  func trainingStepCompleted(step: Int, totalSteps: Int, loss: Float) {
    lastStep = step
    lastLoss = loss
  }

  func trainingCheckpointSaved(step: Int, path: URL) {
    checkpointPaths.append(path)
  }
}
