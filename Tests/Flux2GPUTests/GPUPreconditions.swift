import Foundation
import Metal
// GPUPreconditions.swift — Shared GPU guard for Flux2GPUTests
import Testing

func checkGPUPreconditions(minimumBytes: UInt64) -> Bool {
  guard MTLCreateSystemDefaultDevice() != nil else {
    Issue.record("No Metal device available")
    return false
  }
  guard ProcessInfo.processInfo.physicalMemory >= minimumBytes else {
    Issue.record(
      "Insufficient memory: \(ProcessInfo.processInfo.physicalMemory) bytes, need \(minimumBytes)")
    return false
  }
  return true
}
