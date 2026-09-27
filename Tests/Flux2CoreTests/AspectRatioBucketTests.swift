// AspectRatioBucketTests.swift - Multi-resolution bucket generation and assignment.
//
// Buckets feed the VAE, so every dimension must be a multiple of 64 and never
// exceed the largest configured base resolution. Assignment must pick the
// closest aspect ratio so training doesn't distort images.

import Foundation
import Testing

@testable import Flux2Core

@Suite struct AspectRatioBucketTests {

  // MARK: - Bucket generation

  @Test(arguments: [[512], [512, 768, 1024], [1024, 512]])
  func generatedBucketsAreVAEAlignedAndCapped(resolutions: [Int]) {
    let manager = AspectRatioBucketManager(resolutions: resolutions)
    let maxDim = resolutions.max()!

    #expect(!manager.buckets.isEmpty)
    for bucket in manager.buckets {
      #expect(bucket.width % 64 == 0, "\(bucket.description) width not /64")
      #expect(bucket.height % 64 == 0, "\(bucket.description) height not /64")
      #expect(bucket.width <= maxDim && bucket.height <= maxDim, "\(bucket.description)")
      #expect(bucket.width > 0 && bucket.height > 0)
    }
  }

  @Test func baseResolutionsAreSorted() {
    let manager = AspectRatioBucketManager(resolutions: [1024, 512, 768])
    #expect(manager.baseResolutions == [512, 768, 1024])
  }

  @Test func bucketsAreUniqueAndSortedByPixelCount() {
    let manager = AspectRatioBucketManager(resolutions: [512, 768, 1024])
    let buckets = manager.buckets

    #expect(Set(buckets).count == buckets.count)
    let pixels = buckets.map(\.totalPixels)
    #expect(pixels == pixels.sorted())
  }

  @Test func everyBaseResolutionHasASquareBucket() {
    let manager = AspectRatioBucketManager(resolutions: [512, 768, 1024])
    for res in [512, 768, 1024] {
      #expect(manager.buckets.contains(ResolutionBucket(width: res, height: res)), "\(res)")
    }
  }

  @Test func bucketSetIsSymmetricForPortraitAndLandscape() {
    // Every landscape ratio has its portrait counterpart in standardAspectRatios,
    // so every w×h bucket should have a matching h×w bucket.
    let manager = AspectRatioBucketManager(resolutions: [512, 768])
    let set = Set(manager.buckets)
    for bucket in manager.buckets {
      #expect(
        set.contains(ResolutionBucket(width: bucket.height, height: bucket.width)),
        "missing transpose of \(bucket.description)")
    }
  }

  @Test func resolutionBucketDerivedProperties() {
    let bucket = ResolutionBucket(width: 768, height: 512)
    #expect(bucket.aspectRatio == 1.5)
    #expect(bucket.totalPixels == 393_216)
    #expect(bucket.description == "768x512")
  }

  // MARK: - Best-bucket selection

  @Test func squareImagePicksSquareBucket() {
    let manager = AspectRatioBucketManager(resolutions: [512])
    let bucket = manager.findBestBucket(width: 512, height: 512)
    #expect(bucket == ResolutionBucket(width: 512, height: 512))
  }

  @Test(arguments: [(1920, 1080), (1080, 1920), (3000, 2000), (800, 600)])
  func bestBucketHasClosestAspectRatio(width: Int, height: Int) {
    let manager = AspectRatioBucketManager(resolutions: [1024])
    let imageAR = Float(width) / Float(height)
    let chosen = manager.findBestBucket(width: width, height: height)

    // These images are all ≥ bucket size, so there's no upscale penalty and
    // the choice must be the pure closest-aspect-ratio bucket.
    let minDiff = manager.buckets.map { abs($0.aspectRatio - imageAR) }.min()!
    #expect(abs(chosen.aspectRatio - imageAR) == minDiff)
    // Orientation is preserved.
    #expect((chosen.width >= chosen.height) == (width >= height))
  }

  @Test func tinyImageAvoidsHeavilyUpscaledBuckets() {
    // A 64×64 image at [512, 1024]: every square bucket has arDiff 0, but the
    // 1024 bucket needs a 16× upscale (penalty 7.25) vs 8× for 512 (3.25).
    let manager = AspectRatioBucketManager(resolutions: [512, 1024])
    let bucket = manager.findBestBucket(width: 64, height: 64)
    #expect(bucket.width < 1024)
  }

  // MARK: - Assignment / retrieval

  @Test func assignSampleRecordsBucketAndGroupsSamples() throws {
    let manager = AspectRatioBucketManager(resolutions: [512])
    manager.assignSample(filename: "a.png", caption: "A", originalWidth: 800, originalHeight: 800)
    manager.assignSample(filename: "b.png", caption: "B", originalWidth: 600, originalHeight: 600)
    manager.assignSample(filename: "c.png", caption: "C", originalWidth: 1600, originalHeight: 900)

    let square = ResolutionBucket(width: 512, height: 512)
    #expect(manager.getBucket(for: "a.png") == square)
    #expect(manager.getBucket(for: "b.png") == square)

    let wide = try #require(manager.getBucket(for: "c.png"))
    #expect(wide.width > wide.height)

    #expect(manager.getSamples(in: square).map(\.filename) == ["a.png", "b.png"])
    #expect(manager.getSamples(in: square).map(\.caption) == ["A", "B"])
    #expect(manager.getSamples(in: wide).map(\.filename) == ["c.png"])
    #expect(Set(manager.getNonEmptyBuckets()) == Set([square, wide]))
  }

  @Test func unknownFilenameAndEmptyBucketReturnNothing() {
    let manager = AspectRatioBucketManager(resolutions: [512])
    #expect(manager.getBucket(for: "missing.png") == nil)
    #expect(manager.getSamples(in: ResolutionBucket(width: 512, height: 512)).isEmpty)
    #expect(manager.getNonEmptyBuckets().isEmpty)
  }

  @Test func statisticsReportsCounts() {
    let manager = AspectRatioBucketManager(resolutions: [512])
    manager.assignSample(filename: "a.png", caption: "", originalWidth: 512, originalHeight: 512)
    manager.assignSample(filename: "b.png", caption: "", originalWidth: 512, originalHeight: 512)

    let stats = manager.statistics
    #expect(stats.contains("Total buckets: \(manager.buckets.count)"))
    #expect(stats.contains("Non-empty buckets: 1"))
    #expect(stats.contains("512x512: 2 samples"))
  }

  @Test func clearRemovesAssignmentsButKeepsBuckets() {
    let manager = AspectRatioBucketManager(resolutions: [512])
    let bucketCount = manager.buckets.count
    manager.assignSample(filename: "a.png", caption: "", originalWidth: 512, originalHeight: 512)

    manager.clear()

    #expect(manager.getBucket(for: "a.png") == nil)
    #expect(manager.getNonEmptyBuckets().isEmpty)
    #expect(manager.buckets.count == bucketCount)
  }
}
