// TrainingSessionTests.swift - Public pause/resume/stop façade and resume-mode guards.
//
// TrainingSession wraps TrainingController for GUI hosts. These tests cover
// its control delegation, observer plumbing, and the checkpoint/resume guards
// in `start()` that run *before* any training (so no weights or GPU work are
// needed — the tiny transformer passed in is never evaluated).

import Foundation
import Testing

@testable import Flux2Core

@Suite final class TrainingSessionTests {

  let tempDir: URL

  init() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("TrainingSessionTests_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  deinit {
    try? FileManager.default.removeItem(at: tempDir)
  }

  // MARK: - Helpers

  /// Tiny, never-evaluated transformer to satisfy `start()`'s signature.
  private func tinyTransformer() -> Flux2Transformer2DModel {
    Flux2Transformer2DModel(
      config: Flux2TransformerConfig(
        numLayers: 1,
        numSingleLayers: 1,
        attentionHeadDim: 8,
        numAttentionHeads: 2,
        jointAttentionDim: 16,
        pooledProjectionDim: 16,
        axesDimsRope: [2, 2, 2, 2]
      ))
  }

  private func writeCheckpoint(
    step: Int, modelType: String = Flux2Model.klein4B.rawValue, rank: Int = 32,
    alpha: Float = 32.0
  ) throws {
    let dir = tempDir.appendingPathComponent("checkpoint_\(String(format: "%06d", step))")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var state = TrainingState(
      currentStep: step, totalSteps: 1000, rngSeed: 42, configHash: "h",
      modelType: modelType, loraRank: rank, loraAlpha: alpha)
    // A loss must be recorded first: see freshTrainingStateCannotBeSavedAsJSON.
    state.recordLoss(0.5)
    try state.save(to: dir.appendingPathComponent("training_state.json"))
  }

  private func start(
    _ session: TrainingSession, resumeMode: TrainingSession.ResumeMode
  ) async throws {
    var config = SimpleLoRAConfig(outputDir: tempDir)
    config.rank = 32
    config.alpha = 32.0
    try await session.start(
      config: config,
      modelType: .klein4B,
      resumeMode: resumeMode,
      transformer: tinyTransformer(),
      cachedLatents: [],
      cachedEmbeddings: [:]
    )
  }

  private func isIncompatible(_ error: any Error, containing text: String) -> Bool {
    guard case TrainingSession.SessionError.incompatibleCheckpoint(let reason) = error else {
      return false
    }
    return reason.contains(text)
  }

  // MARK: - Properties

  @Test func freshSessionIsIdleAndExposesOutputDirectory() {
    let session = TrainingSession(outputDirectory: tempDir)
    #expect(session.status == .idle)
    #expect(session.state == nil)
    #expect(session.outputDirectory == tempDir)
  }

  @Test func stateAndStatusReflectController() {
    let session = TrainingSession(outputDirectory: tempDir)
    let state = TrainingState(
      currentStep: 7, totalSteps: 10, rngSeed: 1, configHash: "h",
      modelType: "klein-4b", loraRank: 4, loraAlpha: 4)
    session.controller.updateState(state)
    session.controller.setStatus(.running)

    #expect(session.status == .running)
    #expect(session.state?.currentStep == 7)
  }

  // MARK: - Control delegation

  @Test func pauseSignalsControllerWritesPauseFileAndNotifies() {
    let session = TrainingSession(outputDirectory: tempDir)
    let observer = RecordingObserver()
    session.addObserver(observer)

    session.pause()

    #expect(session.controller.shouldPause())
    #expect(FileManager.default.fileExists(atPath: tempDir.appendingPathComponent(".pause").path))
    #expect(observer.statuses == [.paused])
  }

  @Test func resumeFromPausedReturnsToRunningAndNotifiesStep() {
    let session = TrainingSession(outputDirectory: tempDir)
    let observer = RecordingObserver()
    session.addObserver(observer)
    session.controller.updateState(
      TrainingState(
        currentStep: 42, totalSteps: 100, rngSeed: 1, configHash: "h",
        modelType: "klein-4b", loraRank: 4, loraAlpha: 4))
    session.pause()
    session.controller.setStatus(.paused)

    session.resume()

    #expect(session.status == .running)
    #expect(!session.controller.shouldPause())
    #expect(!FileManager.default.fileExists(atPath: tempDir.appendingPathComponent(".pause").path))
    #expect(observer.resumedSteps == [42])
  }

  @Test func resumeWhenNotPausedDoesNotChangeStatusOrNotify() {
    let session = TrainingSession(outputDirectory: tempDir)
    let observer = RecordingObserver()
    session.addObserver(observer)

    session.resume()

    #expect(session.status == .idle)
    #expect(observer.resumedSteps.isEmpty)
  }

  @Test func stopRequestsGracefulStopOnly() {
    let session = TrainingSession(outputDirectory: tempDir)
    session.stop()

    #expect(session.controller.shouldStop())
    #expect(!session.controller.shouldForceStop())
    #expect(FileManager.default.fileExists(atPath: tempDir.appendingPathComponent(".stop").path))
  }

  @Test func forceStopSetsBothStopFlags() {
    let session = TrainingSession(outputDirectory: tempDir)
    session.forceStop()

    #expect(session.controller.shouldForceStop())
    #expect(session.controller.shouldStop())
  }

  @Test func checkpointRequestIsConsumedOnce() {
    let session = TrainingSession(outputDirectory: tempDir)
    session.checkpoint()

    #expect(session.controller.shouldCheckpoint())
    #expect(!session.controller.shouldCheckpoint())
  }

  @Test func waitIsANoOp() async throws {
    let session = TrainingSession(outputDirectory: tempDir)
    try await session.wait()
    #expect(session.status == .idle)
  }

  // MARK: - Observers

  @Test func removedObserverStopsReceivingEvents() {
    let session = TrainingSession(outputDirectory: tempDir)
    let kept = RecordingObserver()
    let removed = RecordingObserver()
    session.addObserver(kept)
    session.addObserver(removed)

    session.controller.setStatus(.running)
    session.removeObserver(removed)
    session.controller.setStatus(.completed)

    #expect(kept.statuses == [.running, .completed])
    #expect(removed.statuses == [.running])
  }

  @Test func observersAreHeldWeakly() {
    let session = TrainingSession(outputDirectory: tempDir)
    weak var weakObserver: RecordingObserver?
    do {
      let observer = RecordingObserver()
      weakObserver = observer
      session.addObserver(observer)
    }
    #expect(weakObserver == nil)
    // Notifying with only a dead reference must not crash.
    session.controller.setStatus(.running)
  }

  // MARK: - start() guards (all throw before any training work)

  @Test(arguments: [TrainingStatus.running, .paused, .checkpointing, .cancelled])
  func startWhileNotIdleThrowsAlreadyRunning(status: TrainingStatus) async {
    let session = TrainingSession(outputDirectory: tempDir)
    session.controller.setStatus(status)

    await #expect {
      try await self.start(session, resumeMode: .forceFresh)
    } throws: { error in
      guard case TrainingSession.SessionError.alreadyRunning = error else { return false }
      return true
    }
    #expect(session.status == status)
  }

  @Test func freshModeRefusesExistingCheckpoint() async throws {
    try writeCheckpoint(step: 100)
    let session = TrainingSession(outputDirectory: tempDir)

    await #expect {
      try await self.start(session, resumeMode: .fresh)
    } throws: { error in
      guard case TrainingSession.SessionError.invalidState(let reason) = error else {
        return false
      }
      return reason.contains("step 100")
    }
    #expect(session.status == .idle)
  }

  @Test func fromStepWithoutCheckpointThrowsNoCheckpointFound() async throws {
    try writeCheckpoint(step: 100)
    let session = TrainingSession(outputDirectory: tempDir)

    await #expect {
      try await self.start(session, resumeMode: .fromStep(50))
    } throws: { error in
      guard case TrainingSession.SessionError.noCheckpointFound = error else { return false }
      return true
    }
  }

  @Test func fromStepRejectsRankMismatch() async throws {
    try writeCheckpoint(step: 100, rank: 16)
    let session = TrainingSession(outputDirectory: tempDir)

    await #expect {
      try await self.start(session, resumeMode: .fromStep(100))
    } throws: { error in
      self.isIncompatible(error, containing: "LoRA rank mismatch")
    }
  }

  @Test func autoResumeRejectsModelTypeMismatch() async throws {
    try writeCheckpoint(step: 200, modelType: Flux2Model.klein9B.rawValue)
    let session = TrainingSession(outputDirectory: tempDir)

    await #expect {
      try await self.start(session, resumeMode: .autoResume)
    } throws: { error in
      self.isIncompatible(error, containing: "Model type mismatch")
    }
  }

  @Test func autoResumeRejectsAlphaMismatchOnLatestCheckpoint() async throws {
    // Older checkpoint is compatible; the latest (300) is not — latest wins.
    try writeCheckpoint(step: 100)
    try writeCheckpoint(step: 300, alpha: 8.0)
    let session = TrainingSession(outputDirectory: tempDir)

    await #expect {
      try await self.start(session, resumeMode: .autoResume)
    } throws: { error in
      self.isIncompatible(error, containing: "LoRA alpha mismatch")
    }
  }

  /// SUSPECTED BUG (documents current behavior): a TrainingState with no
  /// recorded loss has `bestLoss == .infinity`, which JSONEncoder refuses to
  /// encode, so `save(to:)` throws. A checkpoint written before the first loss
  /// is recorded (e.g. an immediate pause/checkpoint request) would fail.
  @Test func freshTrainingStateCannotBeSavedAsJSON() {
    let state = TrainingState(
      currentStep: 0, totalSteps: 10, rngSeed: 1, configHash: "h",
      modelType: "klein-4b", loraRank: 4, loraAlpha: 4)
    #expect(state.bestLoss == .infinity)
    #expect(throws: EncodingError.self) {
      try state.save(to: self.tempDir.appendingPathComponent("state.json"))
    }
  }

  @Test func sessionErrorDescriptions() {
    typealias E = TrainingSession.SessionError
    #expect(E.alreadyRunning.errorDescription == "Training session is already running")
    #expect(E.notRunning.errorDescription == "Training session is not running")
    #expect(E.noCheckpointFound.errorDescription == "No checkpoint found to resume from")
    #expect(E.incompatibleCheckpoint("x").errorDescription == "Checkpoint is incompatible: x")
    #expect(E.missingOptimizerState.errorDescription == "Optimizer state not found in checkpoint")
    #expect(E.invalidState("y").errorDescription == "Invalid training state: y")
  }
}

// MARK: - Recording observer

private final class RecordingObserver: TrainingObserver {
  var statuses: [TrainingStatus] = []
  var resumedSteps: [Int] = []

  func trainingStatusChanged(_ status: TrainingStatus) {
    statuses.append(status)
  }

  func trainingResumed(atStep: Int) {
    resumedSteps.append(atStep)
  }
}
