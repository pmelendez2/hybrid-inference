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

struct HailoDevice {
    handle: hailort::hailo_device,
}

impl Drop for HailoDevice {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: This handle was successfully created by HailoRT and has
        // not previously been released.
        let status = unsafe { hailort::hailo_release_device(self.handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            eprintln!("hailo_release_device failed: status={status}");
        }
    }
}

fn main() {
    let device_ids = scan_devices();

    let Some(device_id) = device_ids.first() else {
        eprintln!("No Hailo devices discovered");
        std::process::exit(1);
    };

    println!("Opening Hailo device: {}", device_id_to_string(device_id));

    let _device = open_device(device_id);

    println!("Hailo device opened successfully");
    println!("The device will now be released automatically");
}

fn scan_devices() -> Vec<hailort::hailo_device_id_t> {
    let mut device_ids = vec![hailort::hailo_device_id_t::default(); DEVICE_CAPACITY];
    let mut device_count = device_ids.len();

    // SAFETY:
    // - HailoRT 4.23 requires params to be null.
    // - device_ids is writable for DEVICE_CAPACITY entries.
    // - device_count points to a valid usize.
    let status = unsafe {
        hailort::hailo_scan_devices(ptr::null_mut(), device_ids.as_mut_ptr(), &mut device_count)
    };

    if status != hailort::hailo_status_HAILO_SUCCESS {
        eprintln!("hailo_scan_devices failed: status={status}");
        std::process::exit(1);
    }

    if device_count > device_ids.len() {
        eprintln!(
            "HailoRT returned {device_count} devices, but capacity is {}",
            device_ids.len()
        );
        std::process::exit(1);
    }

    device_ids.truncate(device_count);
    device_ids
}

fn open_device(device_id: &hailort::hailo_device_id_t) -> HailoDevice {
    let mut handle: hailort::hailo_device = ptr::null_mut();

    // SAFETY:
    // - device_id points to a valid ID returned by hailo_scan_devices.
    // - &mut handle is writable storage for the returned opaque handle.
    let status = unsafe { hailort::hailo_create_device_by_id(device_id, &mut handle) };

    if status != hailort::hailo_status_HAILO_SUCCESS {
        eprintln!("hailo_create_device_by_id failed: status={status}");
        std::process::exit(1);
    }

    if handle.is_null() {
        eprintln!("HailoRT reported success but returned a null device handle");
        std::process::exit(1);
    }

    HailoDevice { handle }
}

fn device_id_to_string(device: &hailort::hailo_device_id_t) -> String {
    let end = device
        .id
        .iter()
        .position(|character| *character == 0)
        .unwrap_or(device.id.len());

    String::from_utf8_lossy(&device.id[..end]).into_owned()
}
