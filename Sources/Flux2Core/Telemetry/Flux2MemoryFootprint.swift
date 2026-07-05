import Darwin
import Foundation

/// Small helper that captures the process's physical memory footprint
/// (`phys_footprint`, read via `task_info` with the `TASK_VM_INFO` flavor)
/// for attachment to phase-boundary telemetry events.
///
/// Boundaries, not internals (AGENTS.md §11 / CLAUDE.md §5a): this is sampled
/// only at the four phase-boundary emit sites (`weightLoadComplete`,
/// `textEncodeComplete`, `denoiseLoopEnd`, `vaeDecodeComplete`) — never
/// per-step, per-block, or per-attention-head.
///
/// `phys_footprint` is the most accurate on-device measure of "real" memory
/// usage (it matches the value the OS uses for jetsam decisions), which is why
/// it is the metric the FLUX-on-iPad working-set recalibration (Sortie B3)
/// consumes.
public enum Flux2MemoryFootprint {

  /// Current process `phys_footprint` in bytes, or `nil` if the Mach call
  /// fails (e.g. on a platform where `TASK_VM_INFO` is unavailable).
  public static func current() -> Int64? {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)

    let result = withUnsafeMutablePointer(to: &info) { infoPtr in
      infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
      }
    }

    guard result == KERN_SUCCESS else { return nil }
    return Int64(info.phys_footprint)
  }
}
