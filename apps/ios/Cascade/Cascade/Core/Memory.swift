import Darwin
import Foundation
import os

/// The process's memory as jetsam sees it (task_vm_info), and what the OS says is still available to it.
struct MemorySnapshot: Sendable {
    let footprintMB: Double, footprintPeakMB: Double, residentMB: Double, residentPeakMB: Double
    /// os_proc_available_memory(): bytes the app can still allocate before hitting its jetsam limit. 0 where unsupported
    /// (the Simulator and macOS), so nil there.
    let availableMB: Double?

    static func now() -> MemorySnapshot {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        let mb = { (v: UInt64) in Double(v) / 1_048_576 }
        let avail = os_proc_available_memory()
        guard kr == KERN_SUCCESS else { return MemorySnapshot(footprintMB: 0, footprintPeakMB: 0, residentMB: 0, residentPeakMB: 0, availableMB: avail > 0 ? mb(UInt64(avail)) : nil) }
        return MemorySnapshot(footprintMB: mb(info.phys_footprint), footprintPeakMB: mb(UInt64(info.ledger_phys_footprint_peak)),
                              residentMB: mb(info.resident_size), residentPeakMB: mb(info.resident_size_peak),
                              availableMB: avail > 0 ? mb(UInt64(avail)) : nil)
    }

    var dict: [String: Double] {
        var d = ["footprint_mb": footprintMB, "footprint_peak_mb": footprintPeakMB, "resident_mb": residentMB, "resident_peak_mb": residentPeakMB]
        if let availableMB { d["available_mb"] = availableMB }
        return d
    }
}
