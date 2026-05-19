# Testing Requirements: flux-2-swift-mlx

This document defines the testing standard for the `flux-2-swift-mlx` package. It establishes which behaviors must be covered, how tests are structured, how CI differs from local development, and where the current gaps are.

---

## 1. Test Targets

| Target | CI | Local | Requires |
|---|---|---|---|
| **FluxTextEncodersTests** | Yes | Yes | Nothing (no GPU, no downloads) |
| **Flux2CoreTests** | Partial | Yes | Nothing for config tests; GPU + models for inference tests |
| **Flux2GPUTests** *(missing — to be created)* | No | Yes | Apple Silicon, 16 GB RAM, downloaded models |

The existing `Flux2CoreTests` mixes configuration tests (CI-safe) and stubs for inference (not yet written). A separate `Flux2GPUTests` target should be created for all tests requiring actual model weights.

---

## 2. CI Configuration

### Runners and Destinations

| Platform | Runner | Destination |
|---|---|---|
| macOS | `macos-26` (Apple Silicon, arm64) | `platform=macOS,arch=arm64` |
| iOS Simulator | `macos-26` | `platform=iOS Simulator,name=iPhone 17,OS=26.1` |

### Required Status Checks (both must pass before merge to `main` or `development`)

- `Test FluxTextEncoders (macOS)`
- `Test Flux2Core — Config Only (macOS)`

### xcodebuild Flags

```bash
xcodebuild test \
  -scheme FluxTextEncoders \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation \
  ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=YES \
  COMPILER_INDEX_STORE_ENABLE=NO
```

### Timeout

30 minutes maximum per CI job. Any test that takes longer than 5 seconds on CI is either doing I/O it shouldn't, or should be in `Flux2GPUTests`.

---

## 3. Test Framework

All tests use **Swift Testing** (`import Testing`), not XCTest, for consistency with the broader intrusive-memory ecosystem.

```swift
import Testing
@testable import Flux2Core

@Suite("Scheduler") struct SchedulerTests {
    @Test func sigmaCurveIsMonotonicallyDecreasing() { ... }
}
```

Use `#expect()` and `#require()` — not `XCTAssert*`.

---

## 4. What Must Be Tested in CI (No GPU Required)

### 4a. FluxTextEncoders — currently covered, maintain coverage

- All `ModelVariant`, `Qwen3Variant`, `KleinVariant` enum cases enumerate correctly
- `GenerateParameters` presets (greedy, creative, balanced) produce distinct configs
- `HiddenStatesConfig` layer index validation and pooling strategies
- `ModelRegistry` contains all expected variants with non-empty HuggingFace repos
- `TekkenTokenizer` round-trips simple ASCII, Unicode, and special characters
- `ImageProcessor` returns correct dimensions and pixel format (no GPU needed for format validation)
- `TextEncoderMemoryConfig` returns expected values for each model variant
- All error cases have non-nil `localizedDescription`

### 4b. Flux2Core — configuration and math, no model weights

- `Flux2TransformerConfig` defaults and custom init (heads, layers, hidden size)
- `VAEConfig` scaling factors and channel counts
- `Flux2QuantizationConfig` presets: `highQuality`, `balanced`, `minimal`, `ultraMinimal`
- `ModelRegistry` maps all `Flux2Model` cases to non-empty HuggingFace repos and CDN paths
- `FlowMatchEulerScheduler`: timestep array length, first/last sigma values, monotonic decrease
- `LatentUtils.pack()` and `unpack()` round-trip for standard resolutions (512×512, 1024×1024)
- `LatentUtils` rejects invalid dimensions (non-multiples of 16) with a clear error
- Position ID generation for RoPE is deterministic for a given resolution
- `MemoryManager.hasEnoughMemory(for:)` thresholds match documented model specs (Klein 4B: 16 GB, Klein 9B: 24 GB)
- All `AcervoError`-equivalent Flux2 error cases carry context in `localizedDescription`
- `LoRAConfig` YAML round-trip (serialize → deserialize → compare)
- `TrainingState` initializes to zero loss, zero step, `running` status
- `TrainingController` pause/resume/stop transitions are correct
- `AspectRatioBucket` assigns images to correct buckets

---

## 5. What Must Be Tested Locally Only (GPU Required)

Create a `Flux2GPUTests` test target. These tests run locally and on scheduled CI (never on PR CI).

### 5a. Precondition Checks

All GPU tests must check preconditions and **fail with a clear message** — do not skip.

```swift
@Test func klien4BGeneration() async throws {
    guard MTLCreateSystemDefaultDevice() != nil else {
        Issue.record("No Metal GPU — Klein 4B requires Apple Silicon")
        return
    }
    let minimumBytes: UInt64 = 16 * 1_073_741_824
    guard ProcessInfo.processInfo.physicalMemory >= minimumBytes else {
        Issue.record("Insufficient memory: need 16 GB for Klein 4B")
        return
    }
    // ... test body
}
```

### 5b. FluxTextEncoders — model inference

- Load Qwen3-4B-8bit; extract embeddings for a short prompt
- Embedding shape: `[1, 512, 3072]` for Klein 4B extractor
- Embedding values are finite (no NaN, no Inf)
- `generate()` returns non-empty text for a simple prompt
- `extractKleinEmbeddings()` is deterministic for identical input and seed

### 5c. Flux2Core — pipeline integration

- Load Klein 4B transformer weights; verify `isModelLoaded == true`
- Generate a 512×512 image in 4 steps; verify `CGImage` is non-nil
- Output image has correct dimensions (512 × 512)
- VAE decode output has finite pixel values (no black/white clamp artifacts)
- LoRA loading: load a test adapter, verify weight shapes match expected rank
- Progress callback fires `steps` times during generation (step count matches config)
- Cancellation mid-generation does not leak GPU memory or crash
- Image-to-image mode produces output within ±10% brightness of reference
- Fixed seed produces bit-identical output across two runs

### 5d. Timeout Values

| Test | Local Timeout |
|---|---|
| Model load (Klein 4B) | 120 seconds |
| Single generation (4 steps, 512×512) | 180 seconds |
| Full quality generation (28 steps, 1024×1024) | 600 seconds |

---

## 6. Dual-Mode Pattern

Tests that can run in degraded form on CI should detect the `CI` environment variable:

```swift
let isCI = ProcessInfo.processInfo.environment["CI"] != nil
```

- **CI mode**: Run only configuration assertions. Skip any call that requires loading weights.
- **Local mode**: Execute full pipeline with downloaded models.

Do not use `XCTSkip` or `#skip`. If a local precondition is unmet, call `Issue.record(...)` and `return`.

---

## 7. Test Helpers (to be created)

### `MockFlux2Pipeline`

A `Flux2Pipeline` stand-in that:
- Returns a synthetic `CGImage` (opaque 512×512 white square, valid PNG)
- Fires progress callbacks with configurable step count
- Can be configured to throw a specific error
- Used by all unit tests that exercise pipeline orchestration without GPU

```swift
final class MockFlux2Pipeline: @unchecked Sendable {
    var simulatedSteps: Int = 4
    var errorToThrow: Error? = nil
    
    func generate(..., progress: (Int, Int) -> Void) async throws -> CGImage {
        if let error = errorToThrow { throw error }
        for step in 1...simulatedSteps { progress(step, simulatedSteps) }
        return TestImage.make()
    }
}
```

### `TestImage`

Synthesizes a minimal valid `CGImage` without bundled resources:

```swift
enum TestImage {
    static func make(width: Int = 64, height: Int = 64) -> CGImage { ... }
}
```

---

## 8. Coverage Gaps (Priority Order)

| Priority | Gap | Resolution |
|---|---|---|
| 1 | No test verifies VAE encode → latent → decode round-trip | Add to `Flux2GPUTests` |
| 2 | No test verifies embedding shape from `KleinEmbeddingExtractor` | Add to `Flux2GPUTests` |
| 3 | No test verifies progress callback fires correct number of times | Add with `MockFlux2Pipeline` (CI-safe) |
| 4 | No test verifies fixed-seed determinism | Add to `Flux2GPUTests` |
| 5 | No test verifies cancellation safety (no crash, no GPU leak) | Add to `Flux2GPUTests` |
| 6 | No test verifies LoRA weight shapes post-load | Add to `Flux2GPUTests` |
| 7 | Quantization presets not verified end-to-end (only config level) | Add to `Flux2GPUTests` |
| 8 | No error-path tests for corrupted model files | Add to unit tests (mock filesystem) |
| 9 | No concurrency test for parallel embedding extraction | Add to unit tests with mocks |
| 10 | `Flux2CoreTests` not gated separately — config tests may fail if GPU-only tests added | Separate targets now |

---

## 9. File Verification Pattern

Any test writing image output must use isolated temporary storage:

```swift
let outputDir = FileManager.default.temporaryDirectory
    .appendingPathComponent("flux2-test-\(UUID())")
try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: outputDir) }

// After generation:
let pngData = try Data(contentsOf: outputDir.appendingPathComponent("output.png"))
let magic = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
#expect(Data(pngData.prefix(8)) == magic, "Output must be a valid PNG")
#expect(pngData.count > 10_000, "PNG must be larger than 10 KB")
```
