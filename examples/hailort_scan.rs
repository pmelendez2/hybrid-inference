use std::ptr;

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod hailort {
    include!(concat!(env!("OUT_DIR"), "/hailort_bindings.rs"));
}

const DEVICE_CAPACITY: usize = 16;

fn main() {
    let mut device_ids = vec![hailort::hailo_device_id_t::default(); DEVICE_CAPACITY];

    // On input this is the allocated array length. HailoRT changes it to
    // the number of devices actually discovered.
    let mut device_count = device_ids.len();

    // SAFETY:
    //
    // - HailoRT 4.23 requires `params` to be null.
    // - `device_ids` points to writable storage for DEVICE_CAPACITY entries.
    // - `device_count` points to a valid usize, matching C's size_t.
    let status = unsafe {
        hailort::hailo_scan_devices(ptr::null_mut(), device_ids.as_mut_ptr(), &mut device_count)
    };

    if status != hailort::hailo_status_HAILO_SUCCESS {
        eprintln!("hailo_scan_devices failed: status={status}");
        std::process::exit(1);
    }

    if device_count > device_ids.len() {
        eprintln!(
            "HailoRT returned {device_count} devices, but only {} entries were allocated",
            device_ids.len()
        );
        std::process::exit(1);
    }

    device_ids.truncate(device_count);

    println!("Hailo devices discovered: {device_count}");

    for (index, device) in device_ids.iter().enumerate() {
        println!("  [{}] {}", index + 1, device_id_to_string(device));
    }
}

fn device_id_to_string(device: &hailort::hailo_device_id_t) -> String {
    let end = device
        .id
        .iter()
        .position(|character| *character == 0)
        .unwrap_or(device.id.len());

    String::from_utf8_lossy(&device.id[..end]).into_owned()
}
