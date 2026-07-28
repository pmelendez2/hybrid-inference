use std::mem::MaybeUninit;

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod hailort {
    include!(concat!(env!("OUT_DIR"), "/hailort_bindings.rs"));
}

fn main() {
    let mut version = MaybeUninit::<hailort::hailo_version_t>::uninit();

    // SAFETY:
    //
    // - `version.as_mut_ptr()` points to valid writable storage.
    // - HailoRT initializes that storage when it returns HAILO_SUCCESS.
    // - We do not call `assume_init` unless the operation succeeds.
    let status = unsafe { hailort::hailo_get_library_version(version.as_mut_ptr()) };

    if status != hailort::hailo_status_HAILO_SUCCESS {
        eprintln!("hailo_get_library_version failed: status={status}");
        std::process::exit(1);
    }

    // SAFETY:
    //
    // HailoRT returned HAILO_SUCCESS, so `version` has been initialized.
    let version = unsafe { version.assume_init() };

    println!(
        "HailoRT library version: {}.{}.{}",
        version.major, version.minor, version.revision
    );
}
