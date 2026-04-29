---
feature_name: OPERATION MARCHING RELAY
starting_point_commit: c90bfc95f31803b3737617dd04d98222078c8efa
mission_branch: mission/marching-relay/1
iteration: 1
---

# EXECUTION_PLAN.md — flux-2-swift-mlx Testing

Source requirements: `TESTING_REQUIREMENTS.md`
Project root: `/Users/stovak/Projects/flux-2-swift-mlx`

---

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure. Maps to *agentic cycles*, not time.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One agent, one goal, one return. Bounded to fit within a single context window.

> **Work Unit** — A grouping of sorties (package, component, phase). A logical subdivision of the mission.

We deliberately avoid agile/waterfall terminology (sprint, iteration, phase) because those map to **time**. Missions and sorties map to **agentic work cycles**, which have no inherent time dimension.

---

## Mission Overview

Bring the `flux-2-swift-mlx` test suite into full compliance with `TESTING_REQUIREMENTS.md`. This means:

1. Creating shared test helpers (`TestHelpers`: `MockFlux2Pipeline`, `TestImage`) in a dedicated test target.
2. Converting all existing XCTest tests in `FluxTextEncodersTests` to Swift Testing (`import Testing`, `#expect`, `#require`).
3. Auditing and completing coverage gaps in `FluxTextEncodersTests` (13 new CI-safe tests).
4. Converting all existing XCTest tests in `Flux2CoreTests` to Swift Testing.
5. Auditing and completing coverage gaps in `Flux2CoreTests` (19 new CI-safe tests).
6. Creating the missing `Flux2GPUTests` target in `Package.swift` with all GPU-gated tests.
7. Splitting the GitHub Actions CI workflow into two named jobs matching required status check names with correct xcodebuild flags.
8. Updating branch protection rules to require the two new status checks.

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|-------------|
| WU-1: Test Helpers | `Tests/` | 1 | 1 | none |
| WU-2: FluxTextEncodersTests Migration | `Tests/FluxTextEncodersTests/` | 2 | 2 | WU-1 |
| WU-3: FluxTextEncodersTests Coverage Gaps | `Tests/FluxTextEncodersTests/` | 1 | 3 | WU-2 |
| WU-4: Flux2CoreTests Migration | `Tests/Flux2CoreTests/` | 2 | 2 | WU-1 |
| WU-5: Flux2CoreTests Coverage Gaps | `Tests/Flux2CoreTests/` | 1 | 3 | WU-4 |
| WU-6: Flux2GPUTests Target | `Tests/Flux2GPUTests/` | 2 | 3 | WU-1, WU-2, WU-4, WU-7 |
| WU-7: Package.swift & CI | `/` | 1 | 2 | WU-1 |
| WU-8: Branch Protection | `/` | 1 | 4 | WU-7 |

---

## Parallelism Structure

**Critical Path**: Sortie 1 → Sortie 4 → Sortie 5 → Sortie 9 → Sortie 10 (length: 5 sorties, including GPU)

**Parallel Execution Groups**:

- **Group 1** (Layer 1 — sequential):
  - Sortie 1: Test Helpers (supervising agent)

- **Group 2** (Layer 2 — parallel opportunities within supervising agent):
  - Sortie 2: Migrate FTE Part 1 (supervising agent — has build step)
  - Sortie 4: Migrate F2C Part 1 (supervising agent — has build step)
  - Sortie 7: Package.swift & CI (supervising agent — has build step)
  - NOTE: These three have no shared file dependencies and could run in parallel with sub-agents for file editing, but all have build/compile exit criteria requiring the supervising agent. Dispatch in priority order: Sortie 4, then Sortie 2, then Sortie 7.

- **Group 3** (Layer 3 — after Layer 2 completes):
  - Sortie 3: Migrate FTE Part 2 (supervising agent — has build step)
  - Sortie 5: Migrate F2C Part 2 (supervising agent — has build step)
  - Sorties 3 and 5 are independent and could run as parallel sub-agents, but both need build verification — supervising agent only.

- **Group 4** (Layer 3 — after migrations complete):
  - Sortie 6: FTE Coverage Gaps (supervising agent)
  - Sortie 8: F2C Coverage Gaps (supervising agent)
  - Sortie 9: Flux2GPUTests Part 1 (supervising agent — build gate)
  - Sortie 8 is sub-agent eligible for file-writing; build verification done by supervising agent.

- **Group 5** (Layer 4 — sequential):
  - Sortie 10: Flux2GPUTests Part 2 (supervising agent — build gate)
  - Sortie 11: Branch Protection (**sub-agent eligible** — no build, `gh` CLI only)

**Agent Constraints**:
- **Supervising agent**: Handles all sorties with build/compile/test steps (Sorties 1–10)
- **Sub-agents (up to 1)**: Sortie 11 only (branch protection — no build required)

---

## WU-1: Test Helpers

### Sortie 1: Create shared test helper files

**Priority**: 24.75 — Highest priority. Foundation target transitively unblocks 7 of 11 sorties. Must go first.

**Entry criteria**:
- [ ] First sortie — no prerequisites.
- [ ] `/Users/stovak/Projects/flux-2-swift-mlx/Tests/` directory exists.
- [ ] `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` exists (agent must read it before implementing mock).

**Tasks**:
1. Read `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` in full to capture the exact `generate` method signature (parameter labels, types, async/throws annotations).
2. Create `Tests/TestHelpers/TestImage.swift` — implement `enum TestImage` with `static func make(width: Int = 64, height: Int = 64) -> CGImage` using `CGContext` and `CGBitmapInfo.byteOrder32Big`. No bundled resources, no disk I/O. Must be pure in-memory.
3. Create `Tests/TestHelpers/MockFlux2Pipeline.swift` — implement `final class MockFlux2Pipeline: @unchecked Sendable` with:
   - `var simulatedSteps: Int = 4`
   - `var errorToThrow: Error?`
   - `func generate(...)` with the **exact same parameter list** as the real `Flux2Pipeline.generate` (copied from step 1), plus a `progress: (Int, Int) -> Void` callback if not already present in the real signature.
   - Body: if `errorToThrow != nil`, throw it; otherwise call `progress(i, simulatedSteps)` for each step `i` in `1...simulatedSteps`, then return `TestImage.make()`.
4. Add `TestHelpers` as a `.testTarget` in `Package.swift` (approach a — separate test target):
   ```swift
   .testTarget(
       name: "TestHelpers",
       dependencies: ["Flux2Core", "FluxTextEncoders"],
       path: "Tests/TestHelpers"
   )
   ```
   Also add `"TestHelpers"` as a dependency to both `FluxTextEncodersTests` and `Flux2CoreTests` in `Package.swift`.
5. Verify `Package.swift` resolves: `xcodebuild -resolvePackageDependencies -scheme Flux2Swift-Package -clonedSourcePackagesDirPath .spm` exits 0.

**Design decision** (pre-answered): Swift 6.2 supports test targets depending on other test targets. Use approach (a) — a separate `TestHelpers` `.testTarget`. Do NOT add `TestHelpers` to `products`.

**Exit criteria**:
- [ ] `Tests/TestHelpers/TestImage.swift` exists and is non-empty.
- [ ] `Tests/TestHelpers/MockFlux2Pipeline.swift` exists and is non-empty.
- [ ] `MockFlux2Pipeline.generate` parameter list exactly matches `Flux2Pipeline.generate` (verified by side-by-side comparison in agent output).
- [ ] Neither file imports XCTest or uses `XCTAssert*`.
- [ ] `Package.swift` contains a `TestHelpers` testTarget declaration with `path: "Tests/TestHelpers"`.
- [ ] `FluxTextEncodersTests` and `Flux2CoreTests` targets in `Package.swift` both list `"TestHelpers"` as a dependency.
- [ ] `xcodebuild -resolvePackageDependencies -scheme Flux2Swift-Package -clonedSourcePackagesDirPath .spm` exits 0.

---

## WU-2: FluxTextEncodersTests Migration (Part 1)

### Sortie 2: Migrate FluxTextEncodersTests — batch 1 of 2 (3 files)

**Priority**: 13.75 — High. Blocks FTE coverage gaps and GPU FTE tests. Dispatch after Sortie 4 (higher priority).

**Entry criteria**:
- [ ] WU-1 Sortie 1 is COMPLETED.
- [ ] `grep -r "import XCTest" Tests/FluxTextEncodersTests/ | wc -l` returns > 0.

**Tasks**:
Migrate exactly these 3 files in `Tests/FluxTextEncodersTests/`:

1. `FluxTextEncodersTests.swift`
2. `GenerationResultTests.swift`
3. `ProfilerTests.swift`

For each file:
- Replace `import XCTest` with `import Testing`.
- Replace `final class … : XCTestCase` with `struct …`.
- Replace `func testXxx()` with `@Test func xxx()`.
- Replace `@MainActor func testXxx()` with `@Test @MainActor func xxx()`.
- Replace all `XCTAssertEqual(a, b, msg)` with `#expect(a == b, "\(msg)")`.
- Replace all `XCTAssertTrue(expr, msg)` with `#expect(expr, "\(msg)")`.
- Replace all `XCTAssertFalse(expr, msg)` with `#expect(!expr, "\(msg)")`.
- Replace all `XCTAssertNil(expr)` with `#expect(expr == nil)`.
- Replace all `XCTAssertNotNil(expr)` with `#expect(expr != nil)`.
- Replace all `XCTAssertGreaterThan(a, b)` with `#expect(a > b)`.
- Replace all `XCTAssertGreaterThanOrEqual(a, b)` with `#expect(a >= b)`.
- Replace all `XCTAssertLessThan(a, b)` with `#expect(a < b)`.
- Replace `XCTAssertThrowsError(expr)` with `#expect(throws: (any Error).self) { try expr }`.
- Remove `setUp()`/`tearDown()` methods; use Swift Testing `init()` and `deinit` or computed properties.
- Wrap each existing class's tests in a `@Suite("ClassName")` struct.
- Do NOT delete any existing test logic.

**Exit criteria**:
- [ ] `grep "import XCTest" Tests/FluxTextEncodersTests/FluxTextEncodersTests.swift Tests/FluxTextEncodersTests/GenerationResultTests.swift Tests/FluxTextEncodersTests/ProfilerTests.swift` returns no output.
- [ ] `grep "XCTAssert" Tests/FluxTextEncodersTests/FluxTextEncodersTests.swift Tests/FluxTextEncodersTests/GenerationResultTests.swift Tests/FluxTextEncodersTests/ProfilerTests.swift` returns no output.
- [ ] `grep "XCTestCase" Tests/FluxTextEncodersTests/FluxTextEncodersTests.swift Tests/FluxTextEncodersTests/GenerationResultTests.swift Tests/FluxTextEncodersTests/ProfilerTests.swift` returns no output.
- [ ] `xcodebuild build-for-testing -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing FluxTextEncodersTests` exits 0.

---

### Sortie 3: Migrate FluxTextEncodersTests — batch 2 of 2 (6 files)

**Priority**: 13.75 — Same as Sortie 2; must complete before FTE gaps and GPU tests.

**Entry criteria**:
- [ ] Sortie 2 (batch 1 migration) is COMPLETED (build passes with batch 1 migrated).

**Tasks**:
Migrate exactly these 6 files in `Tests/FluxTextEncodersTests/`:

1. `ConfigurationTests.swift`
2. `HiddenStatesConfigTests.swift`
3. `ImageProcessorTests.swift`
4. `ModelRegistryTests.swift`
5. `TextEncoderModelDirectoryTests.swift`
6. `TokenizerTests.swift`

Apply the same migration rules as Sortie 2 (XCTest → Swift Testing API surface replacement). Wrap each class in a `@Suite`. Do not delete any existing test logic.

**Exit criteria**:
- [ ] `grep -r "import XCTest" Tests/FluxTextEncodersTests/` returns no output.
- [ ] `grep -r "XCTAssert" Tests/FluxTextEncodersTests/` returns no output.
- [ ] `grep -r "XCTestCase" Tests/FluxTextEncodersTests/` returns no output.
- [ ] Every test function in `Tests/FluxTextEncodersTests/` is annotated with `@Test`.
- [ ] Every test group is wrapped in a `@Suite`.
- [ ] `xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing FluxTextEncodersTests` exits 0.

---

## WU-3: FluxTextEncodersTests Coverage Gaps

### Sortie 6: Fill FluxTextEncodersTests coverage gaps

**Priority**: 1.75 — Low independent priority; sequential after WU-2 completes. No further dependencies.

**Entry criteria**:
- [ ] Sortie 3 is COMPLETED (all FluxTextEncodersTests migrated to Swift Testing and green).

**Tasks**:
Add the following 13 new `@Test` functions to appropriate files in `Tests/FluxTextEncodersTests/`. Each test must annotate with `@Test(.timeLimit(.seconds(30)))` to enforce the execution time budget (enforced by Swift Testing runtime, not wall clock check).

1. `@Test(.timeLimit(.seconds(30))) func allModelVariantCasesEnumerate()` — asserts `ModelVariant.allCases.count > 0` and each case has a non-empty `rawValue` or string description.
2. `@Test(.timeLimit(.seconds(30))) func allQwen3VariantCasesEnumerate()` — same pattern for `Qwen3Variant`.
3. `@Test(.timeLimit(.seconds(30))) func allKleinVariantCasesEnumerate()` — same pattern for `KleinVariant`.
4. `@Test(.timeLimit(.seconds(30))) func generateParameterPresetsAreDistinct()` — asserts `GenerateParameters.greedy`, `.creative`, and `.balanced` produce structs that differ in at least one field.
5. `@Test(.timeLimit(.seconds(30))) func hiddenStatesConfigLayerIndexValidation()` — verify layer index validation logic; assert invalid indices (e.g., negative, or > maxLayer) are rejected by the validation API.
6. `@Test(.timeLimit(.seconds(30))) func modelRegistryHasNonEmptyRepos()` — iterate all `ModelRegistry` variants; assert each HuggingFace repo string is non-empty and contains "/".
7. `@Test(.timeLimit(.seconds(30))) func tekkenTokenizerRoundTrips()` — round-trip ASCII ("hello world"), Unicode ("日本語"), and a special character ("</s>") through encode→decode; assert equality.
8. `@Test(.timeLimit(.seconds(30))) func imageProcessorReturnsDimensions()` — create a synthetic `CGImage` input via `TestImage.make()`; assert returned width and height are positive integers.
9. `@Test(.timeLimit(.seconds(30))) func textEncoderMemoryConfigValues()` — for each model variant, assert `TextEncoderMemoryConfig` returns a non-zero byte count.
10. `@Test(.timeLimit(.seconds(30))) func allErrorCasesHaveLocalizedDescription()` — for every `FluxEncoderError` case, assert `localizedDescription` is non-nil and non-empty.
11. `@Test(.timeLimit(.seconds(30))) func progressCallbackFiresCorrectNumberOfTimes()` — instantiate `MockFlux2Pipeline(simulatedSteps: 6)`; call `generate`; assert callback fired exactly 6 times. (CI-safe: no GPU.)
12. `@Test(.timeLimit(.seconds(30))) func errorPathForCorruptedModelFile()` — configure `MockFlux2Pipeline` with `errorToThrow = NSError(domain: "test", code: 1)`; call `generate`; assert it throws and the thrown error's `localizedDescription` is non-empty.
13. `@Test(.timeLimit(.seconds(30))) func concurrencyParallelEmbeddingExtraction()` — launch two `MockFlux2Pipeline` instances' `generate` calls concurrently with `async let`; assert both complete without throwing.

**Exit criteria**:
- [ ] All 13 new tests are present in `Tests/FluxTextEncodersTests/` (verify with `grep -r "@Test" Tests/FluxTextEncodersTests/ | wc -l` — count must have increased by 13 from pre-sortie baseline).
- [ ] No new test uses `XCTAssert*` or `XCTSkip`.
- [ ] No new test uses `#skip`.
- [ ] Every new test has `@Test(.timeLimit(.seconds(30)))`.
- [ ] `xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing FluxTextEncodersTests` exits 0.

---

## WU-4: Flux2CoreTests Migration

### Sortie 4: Migrate Flux2CoreTests — batch 1 of 2 (main file)

**Priority**: 14.25 — Highest in Layer 2. Dispatch first among Layer 2 sorties. The main test file is 2813 lines; this sortie handles it alone.

**Entry criteria**:
- [ ] WU-1 Sortie 1 is COMPLETED.
- [ ] `grep "import XCTest" Tests/Flux2CoreTests/Flux2CoreTests.swift` returns output.

**Tasks**:
Migrate exactly this 1 file: `Tests/Flux2CoreTests/Flux2CoreTests.swift` (2813 lines).

Apply the same migration rules as Sortie 2 (XCTest → Swift Testing). Additionally:

- For any test that calls GPU or model-loading APIs (e.g., `MLX.eval`, model weight loading, `MTLCreateSystemDefaultDevice`): wrap the GPU call with:
  ```swift
  let isCI = ProcessInfo.processInfo.environment["CI"] != nil
  if isCI { return }
  ```
  Do NOT remove the test; gate only the GPU call.
- Remove any `setUp()`/`tearDown()` XCTest lifecycle methods; use Swift Testing `init()` and `deinit` or computed properties.
- Wrap existing classes in `@Suite` structs.

**Exit criteria**:
- [ ] `grep "import XCTest" Tests/Flux2CoreTests/Flux2CoreTests.swift` returns no output.
- [ ] `grep "XCTAssert" Tests/Flux2CoreTests/Flux2CoreTests.swift` returns no output.
- [ ] `grep "XCTestCase" Tests/Flux2CoreTests/Flux2CoreTests.swift` returns no output.
- [ ] `xcodebuild build-for-testing -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing Flux2CoreTests` exits 0.

---

### Sortie 5: Migrate Flux2CoreTests — batch 2 of 2 (3 remaining files)

**Priority**: 14.25 — Same priority as Sortie 4; sequential within WU-4.

**Entry criteria**:
- [ ] Sortie 4 (batch 1 migration) is COMPLETED (build passes with Flux2CoreTests.swift migrated).

**Tasks**:
Migrate exactly these 3 files:

1. `Tests/Flux2CoreTests/ImageToImageTrainingTests.swift` (367 lines)
2. `Tests/Flux2CoreTests/ModelDirectoryTests.swift` (153 lines)
3. `Tests/Flux2CoreTests/TrainingControlTests.swift` (347 lines)

Apply the same migration rules as Sortie 4 (XCTest → Swift Testing, GPU gating with CI check, `@Suite` wrapping, `setUp`/`tearDown` replacement).

**Exit criteria**:
- [ ] `grep -r "import XCTest" Tests/Flux2CoreTests/` returns no output.
- [ ] `grep -r "XCTAssert" Tests/Flux2CoreTests/` returns no output.
- [ ] `grep -r "XCTestCase" Tests/Flux2CoreTests/` returns no output.
- [ ] Every test function in `Tests/Flux2CoreTests/` is annotated with `@Test`.
- [ ] `xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing Flux2CoreTests` exits 0.

---

## WU-5: Flux2CoreTests Coverage Gaps

### Sortie 8: Fill Flux2CoreTests coverage gaps (CI-safe)

**Priority**: 1.75 — Low independent priority; sequential after WU-4 completes.

**Entry criteria**:
- [ ] Sortie 5 is COMPLETED (Flux2CoreTests migrated and green).

**Tasks**:
Add the following 19 new `@Test(.timeLimit(.seconds(30)))` functions to appropriate files in `Tests/Flux2CoreTests/`:

1. `@Test(.timeLimit(.seconds(30))) func flux2TransformerConfigDefaults()` — assert default head count, layer count, and hidden size match documented values from `Flux2Config.swift`.
2. `@Test(.timeLimit(.seconds(30))) func flux2TransformerConfigCustomInit()` — construct with non-default values; assert all fields are stored.
3. `@Test(.timeLimit(.seconds(30))) func vaeConfigScalingFactors()` — assert scaling factor and channel counts from `VAEConfig.flux2Dev` match expected values.
4. `@Test(.timeLimit(.seconds(30))) func quantizationPresetsAreDistinct()` — assert `highQuality`, `balanced`, `minimal`, `ultraMinimal` all differ from each other in at least one field.
5. `@Test(.timeLimit(.seconds(30))) func modelRegistryMapsAllCases()` — iterate all `Flux2Model` cases; assert HuggingFace repo and CDN path strings are both non-empty.
6. `@Test(.timeLimit(.seconds(30))) func flowMatchSchedulerTimestepCount()` — construct scheduler with a given step count N; assert `timesteps.count == N`.
7. `@Test(.timeLimit(.seconds(30))) func flowMatchSchedulerFirstLastSigma()` — assert `sigmas.first! > sigmas.last!` (monotonic decrease).
8. `@Test(.timeLimit(.seconds(30))) func latentUtilsRoundTrip512()` — `pack` then `unpack` a zeroed tensor at 512×512; assert shape is restored.
9. `@Test(.timeLimit(.seconds(30))) func latentUtilsRoundTrip1024()` — same for 1024×1024.
10. `@Test(.timeLimit(.seconds(30))) func latentUtilsRejectsInvalidDimensions()` — call `pack` with a dimension not a multiple of 16; assert it throws a non-nil error.
11. `@Test(.timeLimit(.seconds(30))) func ropePositionIdsAreDeterministic()` — call position ID generation twice for the same resolution; assert results are equal.
12. `@Test(.timeLimit(.seconds(30))) func memoryManagerKlein4BThreshold()` — assert `MemoryManager.hasEnoughMemory(for: .klein4B)` threshold is 16 GB (16 * 1_073_741_824 bytes).
13. `@Test(.timeLimit(.seconds(30))) func memoryManagerKlein9BThreshold()` — assert threshold for Klein 9B is 24 GB.
14. `@Test(.timeLimit(.seconds(30))) func allFlux2ErrorCasesHaveContext()` — for every error case in the Flux2 error enum, assert `localizedDescription` is non-nil and non-empty.
15. `@Test(.timeLimit(.seconds(30))) func loraConfigYamlRoundTrip()` — serialize a `LoRAConfig` to YAML, deserialize back, compare equality.
16. `@Test(.timeLimit(.seconds(30))) func trainingStateInitializesToZero()` — assert `TrainingState()` has zero loss, zero step, and `.running` status.
17. `@Test(.timeLimit(.seconds(30))) func trainingControllerPauseResume()` — call `pause()`; assert state is `.paused`; call `resume()`; assert state is `.running`.
18. `@Test(.timeLimit(.seconds(30))) func trainingControllerStop()` — call `stop()`; assert state is `.stopped`.
19. `@Test(.timeLimit(.seconds(30))) func aspectRatioBucketAssignment()` — provide several known image sizes; assert each maps to the expected bucket.

**Exit criteria**:
- [ ] All 19 new tests are present in `Tests/Flux2CoreTests/` (verify with `grep -r "@Test" Tests/Flux2CoreTests/ | wc -l` — count must have increased by 19).
- [ ] No new test uses `XCTAssert*`, `#skip`, or `XCTSkip`.
- [ ] Every new test has `@Test(.timeLimit(.seconds(30)))`.
- [ ] `xcodebuild test -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing Flux2CoreTests` exits 0.

---

## WU-6: Flux2GPUTests Target

### Sortie 9: Create Flux2GPUTests target and FluxTextEncoders GPU tests

**Priority**: 6.75 — Medium. External API + GPU risk. Requires all migrations and Package.swift to be complete.

**Entry criteria**:
- [ ] WU-2, WU-4, and WU-7 are all COMPLETED.
- [ ] `Tests/Flux2GPUTests/` does NOT yet exist.
- [ ] `grep "Flux2GPUTests" Package.swift` returns output (WU-7 Sortie 7 has already added the target declaration).

**Tasks**:
1. Create directory `Tests/Flux2GPUTests/`.
2. Create `Tests/Flux2GPUTests/GPUPreconditions.swift` — implement a free function:
   ```swift
   func checkGPUPreconditions(minimumBytes: UInt64) -> Bool {
       guard MTLCreateSystemDefaultDevice() != nil else {
           Issue.record("No Metal device available")
           return false
       }
       guard ProcessInfo.processInfo.physicalMemory >= minimumBytes else {
           Issue.record("Insufficient memory: \(ProcessInfo.processInfo.physicalMemory) bytes, need \(minimumBytes)")
           return false
       }
       return true
   }
   ```
3. Create `Tests/Flux2GPUTests/FluxTextEncodersGPUTests.swift` with `@Suite("FluxTextEncoders GPU")` containing exactly 4 `@Test` functions:
   - `@Test(.timeLimit(.minutes(3))) func qwen3EmbeddingShape()` — load Qwen3-4B-8bit; extract embeddings; assert shape is `[1, 512, 3072]`.
   - `@Test(.timeLimit(.minutes(3))) func embeddingValuesAreFinite()` — assert no NaN or Inf in extracted embeddings.
   - `@Test(.timeLimit(.minutes(3))) func generateReturnsNonEmptyText()` — call `generate()` with a short prompt; assert result is non-empty.
   - `@Test(.timeLimit(.minutes(3))) func kleinEmbeddingsAreDeterministic()` — call `extractKleinEmbeddings()` twice with the same input and seed; assert results are equal.
4. Each test in step 3 must call `checkGPUPreconditions(minimumBytes: 16 * 1_073_741_824)` at the top and `return` (not throw) if it returns `false`.

**Exit criteria**:
- [ ] `Tests/Flux2GPUTests/GPUPreconditions.swift` exists and contains `func checkGPUPreconditions`.
- [ ] `Tests/Flux2GPUTests/FluxTextEncodersGPUTests.swift` exists.
- [ ] `grep "@Test" Tests/Flux2GPUTests/FluxTextEncodersGPUTests.swift | wc -l` returns 4.
- [ ] Every GPU test calls `checkGPUPreconditions` (verify with `grep -c "checkGPUPreconditions" Tests/Flux2GPUTests/FluxTextEncodersGPUTests.swift` returns 4).
- [ ] No GPU test uses `XCTAssert*`, `#skip`, or `XCTSkip`.
- [ ] `xcodebuild build-for-testing -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing Flux2GPUTests` exits 0 (build only — no model download required for build gate).

---

### Sortie 10: Add Flux2Core GPU pipeline integration tests

**Priority**: 4.25 — GPU risk, sequential after Sortie 9.

**Entry criteria**:
- [ ] Sortie 9 is COMPLETED (`Tests/Flux2GPUTests/` exists, `GPUPreconditions.swift` is in place, build passes).
- [ ] `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` has been read by the agent to verify `CancellationError` propagation behavior.

**Tasks**:
1. Read `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift` to determine whether `Task.cancel()` propagates as `CancellationError` directly or wrapped.
2. Create `Tests/Flux2GPUTests/Flux2CoreGPUTests.swift` with `@Suite("Flux2Core GPU")` containing exactly 11 `@Test` functions:
   - `@Test(.timeLimit(.minutes(2))) func klein4BModelLoads()` — load Klein 4B weights; assert `isModelLoaded == true`.
   - `@Test(.timeLimit(.minutes(3))) func generate512x512In4Steps()` — generate 512×512 in 4 steps; assert `CGImage` is non-nil and has `width == 512`, `height == 512`.
   - `@Test(.timeLimit(.minutes(3))) func vaeDecodeHasFinitePixels()` — assert no black/white clamp artifacts (all channel values in `[0.0, 1.0]`).
   - `@Test(.timeLimit(.minutes(3))) func vaeRoundTripEncodeLatentDecode()` — encode a test image to latent space; decode back; assert output is non-nil.
   - `@Test(.timeLimit(.minutes(3))) func kleinEmbeddingExtractorShape()` — assert `KleinEmbeddingExtractor` output shape matches `[1, 512, 3072]`.
   - `@Test(.timeLimit(.minutes(3))) func fixedSeedIsDeterministic()` — run generation twice with identical fixed seed; assert outputs are bit-identical.
   - `@Test(.timeLimit(.minutes(3))) func cancellationDoesNotCrash()` — start generation; cancel mid-flight via `Task.cancel()`; assert no crash and only `CancellationError` is thrown (or no error, depending on pipeline behavior determined in step 1).
   - `@Test(.timeLimit(.minutes(3))) func loraWeightShapesMatchRank()` — synthesize a minimal test LoRA adapter programmatically (random weights saved to `FileManager.default.temporaryDirectory.appendingPathComponent("test-lora-\(UUID())")` with `defer { try? FileManager.default.removeItem(...) }`); load it; assert all weight tensors have the expected rank dimension.
   - `@Test(.timeLimit(.minutes(10))) func quantizationPresetEndToEnd()` — generate with `ultraMinimal` preset; assert output image is non-nil.
   - `@Test(.timeLimit(.minutes(3))) func imageToImageOutputIsNonTrivial()` — run img2img with `TestImage.make()`; assert output brightness is in range `[0.05, 0.95]` (mean pixel value, confirming non-saturated, non-trivial output). NOTE: Specific reference brightness removed — test verifies the output is non-trivial (not all-black, not all-white).
   - `@Test(.timeLimit(.minutes(3))) func progressCallbackFiresStepsTimes()` — run 4-step generation; assert progress callback fires exactly 4 times.
3. Each test must call `checkGPUPreconditions(minimumBytes: 16 * 1_073_741_824)` and `return` on failure.
4. Any test writing image output to disk must use `FileManager.default.temporaryDirectory.appendingPathComponent("flux2-test-\(UUID())")` and `defer { try? FileManager.default.removeItem(...) }`.
5. PNG magic byte validation (`0x89 0x50 0x4E 0x47 ...`) is required for any test that saves and reads back a PNG.

**Design decision** (pre-answered): LoRA test adapter must be synthesized programmatically (no pre-existing test adapter confirmed in repo). Use random weight tensors, serialize to temp directory.

**Exit criteria**:
- [ ] `Tests/Flux2GPUTests/Flux2CoreGPUTests.swift` exists.
- [ ] `grep "@Test" Tests/Flux2GPUTests/Flux2CoreGPUTests.swift | wc -l` returns 11.
- [ ] Every test calls `checkGPUPreconditions` (verify with `grep -c "checkGPUPreconditions" Tests/Flux2GPUTests/Flux2CoreGPUTests.swift` returns 11).
- [ ] Tests writing image output use the temp directory pattern with `defer` cleanup.
- [ ] `xcodebuild build-for-testing -scheme Flux2Swift-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm -only-testing Flux2GPUTests` exits 0.

---

## WU-7: Package.swift & CI Workflow

### Sortie 7: Update Package.swift and GitHub Actions workflow

**Priority**: 8.75 — Medium-high. Enables GPU test target and CI job split. Dispatch in Layer 2 (parallel with WU-2/WU-4 migrations).

**Entry criteria**:
- [ ] WU-1 Sortie 1 is COMPLETED (TestHelpers files exist so Package.swift can reference them — already added in Sortie 1).
- [ ] `Package.swift` currently has no `Flux2GPUTests` target (verify: `grep "Flux2GPUTests" Package.swift` returns no output).
- [ ] `.github/workflows/tests.yml` exists.

**Tasks**:
1. Read `Package.swift` in full.
2. Add a `Flux2GPUTests` test target to `Package.swift`:
   ```swift
   .testTarget(
       name: "Flux2GPUTests",
       dependencies: ["Flux2Core", "FluxTextEncoders", "TestHelpers"],
       path: "Tests/Flux2GPUTests"
   )
   ```
   NOTE: Do NOT add `Flux2GPUTests` to `products`.
3. Read `.github/workflows/tests.yml` in full.
4. Rewrite `.github/workflows/tests.yml` to have exactly two top-level jobs with these **exact** names (these strings are the required status check names):
   - `Test FluxTextEncoders (macOS)`
   - `Test Flux2Core — Config Only (macOS)`
5. Both jobs must:
   - Run on `runs-on: macos-26`.
   - Resolve dependencies: `xcodebuild -resolvePackageDependencies -scheme Flux2Swift-Package -clonedSourcePackagesDirPath .spm`
   - Build and test with flags: `-skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=YES COMPILER_INDEX_STORE_ENABLE=NO -clonedSourcePackagesDirPath .spm`
   - Job 1 uses: `-only-testing FluxTextEncodersTests`
   - Job 2 uses: `-only-testing Flux2CoreTests`
   - Neither job references `Flux2GPUTests`.
6. Preserve the existing `on: pull_request: branches: [main, development]` trigger.

**Exit criteria**:
- [ ] `grep "Flux2GPUTests" Package.swift` returns output (target declared).
- [ ] `xcodebuild -resolvePackageDependencies -scheme Flux2Swift-Package -clonedSourcePackagesDirPath .spm` exits 0.
- [ ] `grep 'name:' .github/workflows/tests.yml | grep -c 'macOS'` returns 2.
- [ ] `grep "Test FluxTextEncoders (macOS)" .github/workflows/tests.yml` returns output.
- [ ] `grep "Test Flux2Core" .github/workflows/tests.yml` returns output.
- [ ] `grep "Flux2GPUTests" .github/workflows/tests.yml` returns no output.
- [ ] Both jobs in `.github/workflows/tests.yml` contain `-skipPackagePluginValidation` and `ARCHS=arm64`.
- [ ] `runs-on: macos-26` appears in both jobs.

---

## WU-8: Branch Protection

### Sortie 11: Update branch protection rules for main and development

**Priority**: 3.5 — Low. External `gh` CLI calls only. **Sub-agent eligible** (no build required).

**Agent**: Sub-agent (no build operations required).

**Entry criteria**:
- [ ] WU-7 Sortie 7 is COMPLETED and the workflow changes are merged to the active branch (job names must be live in remote before protection rules reference them).
- [ ] `gh auth status` exits 0 (authenticated).

**Tasks**:
1. Verify repo identity: `gh repo view --json nameWithOwner` — confirm repo is `intrusive-memory/flux-2-swift-mlx`.
2. Check current branch protection on `main`: `gh api repos/intrusive-memory/flux-2-swift-mlx/branches/main/protection`.
3. Check current branch protection on `development`: same command for `development`.
4. Update `main` branch protection:
   ```bash
   gh api --method PUT repos/intrusive-memory/flux-2-swift-mlx/branches/main/protection \
     --input - <<'EOF'
   {
     "required_status_checks": {
       "strict": true,
       "contexts": ["Test FluxTextEncoders (macOS)", "Test Flux2Core — Config Only (macOS)"]
     },
     "enforce_admins": false,
     "required_pull_request_reviews": null,
     "restrictions": null
   }
   EOF
   ```
5. Repeat for `development` branch.
6. Verify both: `gh api repos/intrusive-memory/flux-2-swift-mlx/branches/main/protection | jq '.required_status_checks.contexts'`

**Exit criteria**:
- [ ] `gh api repos/intrusive-memory/flux-2-swift-mlx/branches/main/protection | jq '.required_status_checks.contexts'` outputs a JSON array containing `"Test FluxTextEncoders (macOS)"` and `"Test Flux2Core — Config Only (macOS)"`.
- [ ] Same for `development` branch.
- [ ] `gh api repos/intrusive-memory/flux-2-swift-mlx/branches/main/protection | jq '.required_status_checks.strict'` returns `true`.
- [ ] Same for `development` branch.

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 8 |
| Total sorties | 11 |
| Dependency structure | 4 layers |
| CI-safe sorties | 1, 2, 3, 4, 5, 6, 7, 8, 11 |
| GPU-only sorties | 9, 10 |
| Framework migration sorties | 2, 3, 4, 5 |
| New test count (CI) | 32 (13 in WU-3 + 19 in WU-5) |
| New test count (GPU) | 15 (4 in Sortie 9 + 11 in Sortie 10) |
| Sub-agent eligible sorties | 11 |

---

## Execution Order (by Layer)

```
Layer 1 (no dependencies):
  Sortie 1 — Test Helpers (creates TestHelpers target, adds to Package.swift)

Layer 2 (requires Layer 1) — dispatch in priority order:
  Sortie 4 — Flux2CoreTests Migration Part 1    [priority 14.25]
  Sortie 2 — FluxTextEncodersTests Migration Part 1  [priority 13.75]
  Sortie 7 — Package.swift & CI                 [priority 8.75]

Layer 3 (sequential within streams):
  Sortie 5 — Flux2CoreTests Migration Part 2    [after Sortie 4]
  Sortie 3 — FluxTextEncodersTests Migration Part 2  [after Sortie 2]
  [After Sortie 5]: Sortie 8 — F2C Coverage Gaps
  [After Sortie 3]: Sortie 6 — FTE Coverage Gaps
  [After Sortie 3 + Sortie 5 + Sortie 7]: Sortie 9 — Flux2GPUTests Part 1

Layer 4 (requires Layer 3 GPU):
  Sortie 10 — Flux2Core GPU pipeline tests      [after Sortie 9]
  Sortie 11 — Branch Protection                 [after Sortie 7 merged, sub-agent]
```

---

## Open Questions & Missing Documentation

All open questions have been resolved or auto-fixed during refinement:

| Sortie | Issue Type | Resolution |
|--------|-----------|------------|
| Sortie 1 | Open question (TestHelpers strategy) | **Resolved**: Use approach (a) — separate `.testTarget(name: "TestHelpers")`. Swift 6.2 supports test-to-test target dependencies. |
| Sortie 1 | Open question (MockFlux2Pipeline signature) | **Addressed in entry criteria**: Agent must read `Flux2Pipeline.swift` before implementing mock. |
| Sortie 7 | Open question (GitHub org name) | **Addressed in tasks**: `gh repo view --json nameWithOwner` added as first task step. |
| Sortie 10 | Open question (LoRA test adapter) | **Resolved**: Synthesize adapter programmatically with random weights + temp directory pattern. Added to task description. |
| Sortie 10 | Open question (CancellationError handling) | **Addressed in tasks**: Read `Flux2Pipeline.swift` first (step 1 of tasks) to determine propagation behavior. |
| Sortie 10 | Vague criterion (img2img brightness reference) | **Auto-fixed**: Changed to assert output brightness in `[0.05, 0.95]` (non-trivial output), removing undefined reference value. |
| Sortie 6, 8 | Vague criterion (wall clock timing) | **Auto-fixed**: Replaced with `@Test(.timeLimit(.seconds(30)))` annotation (enforced by Swift Testing runtime). |
| Sortie 7 | Vague task (TestHelpers approach) | **Auto-fixed**: Approach (a) mandated; no ambiguous decision left to agent. |
| Sortie 9 | Vague entry criterion (Sortie 8 "complete") | **Auto-fixed**: Changed to machine-verifiable: `grep "Flux2GPUTests" Package.swift` returns output. |

**Unresolved items requiring manual review**: None.

---

## Refinement Pass Summary

| Pass | Status | Changes |
|------|--------|---------|
| 1. Atomicity & Testability | PASS | 2 sorties split (Sortie 2 → 2a/2b; Sortie 4 → 4a/4b); renumbered to 11 sorties; 3 vague criteria fixed |
| 2. Prioritization | PASS | Priority scores added to all sorties; Layer 2 dispatch order specified (4, 2, 7); no layer adjustments needed |
| 3. Parallelism | PASS | 1 sub-agent eligible sortie identified (Sortie 11); build constraints enforced; supervising agent handles sorties 1-10 |
| 4. Open Questions & Vague Criteria | PASS | 9 issues found, all auto-fixed or pre-resolved; 0 require manual review |

**VERDICT**: Plan is ready to execute.

### Execution Summary
- Total sorties: 11
- Context budget: 50 turns per sortie
- Largest sortie: Sortie 4 (~48 turns estimated, within 80% budget)
- Critical path length: 5 sorties (1 → 4 → 5 → 9 → 10)
- Parallelism: 1 supervising agent + 1 sub-agent (Sortie 11 only)

**Next step**: `/mission-supervisor start`
