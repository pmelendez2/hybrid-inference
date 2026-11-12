use std::ffi::CString;
use std::fmt;
use std::ptr;

use super::hailort;

#[derive(Debug)]
pub(super) enum RuntimeError {
    GroupIdContainsNullByte,
    HefPathContainsNullByte {
        path: String,
    },
    CreateHefFailed {
        path: String,
        status: hailort::hailo_status,
    },
    NullHefHandle {
        path: String,
    },
    NetworkGroupNameContainsNullByte {
        network_group: String,
    },
    HailoRtCallFailed {
        operation: &'static str,
        status: hailort::hailo_status,
    },
    NullHandle {
        operation: &'static str,
    },
    CountExceedsCapacity {
        resource: &'static str,
        count: usize,
        capacity: usize,
    },
    InvalidVStreamIndex {
        direction: &'static str,
        index: usize,
    },
    BufferSizeMismatch {
        direction: &'static str,
        index: usize,
        actual: usize,
        expected: usize,
    },
    NoConfiguredNetworkGroups,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GroupIdContainsNullByte => write!(f, "VDevice group ID contains a NUL byte"),
            Self::HefPathContainsNullByte { path } => {
                write!(f, "HEF path contains a null byte: {path}")
            }
            Self::CreateHefFailed { path, status } => {
                write!(
                    f,
                    "hailo_create_hef_file failed for {path}: status={status}"
                )
            }
            Self::NullHefHandle { path } => {
                write!(
                    f,
                    "HailoRT returned success but supplied a null HEF handle: {path}"
                )
            }
            Self::NetworkGroupNameContainsNullByte { network_group } => {
                write!(
                    f,
                    "network-group name contains a null byte: {network_group}"
                )
            }
            Self::HailoRtCallFailed { operation, status } => {
                write!(f, "{operation} failed: status={status}")
            }
            Self::NullHandle { operation } => {
                write!(f, "{operation} succeeded but returned a null handle")
            }
            Self::CountExceedsCapacity {
                resource,
                count,
                capacity,
            } => {
                write!(
                    f,
                    "HailoRT reported {count} {resource}, but the buffer capacity is {capacity}"
                )
            }
            Self::InvalidVStreamIndex { direction, index } => {
                write!(f, "invalid {direction} vstream index {index}")
            }
            Self::BufferSizeMismatch {
                direction,
                index,
                actual,
                expected,
            } => {
                write!(
                    f,
                    "{direction} vstream {index} buffer is {actual} bytes; expected {expected} bytes"
                )
            }
            Self::NoConfiguredNetworkGroups => {
                write!(f, "no configured network groups were returned")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

pub(super) struct HailoHef {
    handle: hailort::hailo_hef,
}

impl HailoHef {
    pub(super) fn load(path: &str) -> Result<Self, RuntimeError> {
        let c_path = CString::new(path).map_err(|_| RuntimeError::HefPathContainsNullByte {
            path: path.to_owned(),
        })?;

        let mut handle: hailort::hailo_hef = ptr::null_mut();

        // SAFETY:
        // - c_path is a valid, null-terminated C string.
        // - &mut handle is writable storage for the returned HEF handle.
        // - HailoRT retains the loaded HEF, not the path pointer.
        let status = unsafe { hailort::hailo_create_hef_file(&mut handle, c_path.as_ptr()) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::CreateHefFailed {
                path: path.to_owned(),
                status,
            });
        }

        if handle.is_null() {
            return Err(RuntimeError::NullHefHandle {
                path: path.to_owned(),
            });
        }

        Ok(Self { handle })
    }

    pub(super) fn network_groups(
        &self,
    ) -> Result<Vec<hailort::hailo_network_group_info_t>, RuntimeError> {
        const CAPACITY: usize = 16;

        let mut infos = vec![hailort::hailo_network_group_info_t::default(); CAPACITY];
        let mut count = infos.len();

        // SAFETY:
        // - self.handle is a valid HEF handle owned by this object.
        // - infos provides writable storage for CAPACITY entries.
        // - count initially contains the buffer capacity and is writable.
        let status = unsafe {
            hailort::hailo_get_network_groups_infos(self.handle, infos.as_mut_ptr(), &mut count)
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_get_network_groups_infos",
                status,
            });
        }

        if count > infos.len() {
            return Err(RuntimeError::CountExceedsCapacity {
                resource: "network groups",
                count,
                capacity: infos.len(),
            });
        }

        infos.truncate(count);
        Ok(infos)
    }

    pub(super) fn vstream_infos(
        &self,
        network_group_name: &str,
    ) -> Result<Vec<hailort::hailo_vstream_info_t>, RuntimeError> {
        const CAPACITY: usize = 64;

        let name = CString::new(network_group_name).map_err(|_| {
            RuntimeError::NetworkGroupNameContainsNullByte {
                network_group: network_group_name.to_owned(),
            }
        })?;

        let mut infos = vec![hailort::hailo_vstream_info_t::default(); CAPACITY];
        let mut count = infos.len();

        // SAFETY:
        // - self.handle is a valid HEF handle.
        // - name is a valid null-terminated C string.
        // - infos contains writable storage for CAPACITY entries.
        // - count initially contains the capacity and is writable.
        let status = unsafe {
            hailort::hailo_hef_get_all_vstream_infos(
                self.handle,
                name.as_ptr(),
                infos.as_mut_ptr(),
                &mut count,
            )
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_hef_get_all_vstream_infos",
                status,
            });
        }

        if count > infos.len() {
            return Err(RuntimeError::CountExceedsCapacity {
                resource: "virtual streams",
                count,
                capacity: infos.len(),
            });
        }

        infos.truncate(count);
        Ok(infos)
    }
}

impl Drop for HailoHef {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: The handle was created successfully by HailoRT and has not
        // previously been released.
        let status = unsafe { hailort::hailo_release_hef(self.handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            eprintln!("hailo_release_hef failed: status={status}");
        }
    }
}

struct HailoDevice {
    handle: hailort::hailo_device,
}

impl HailoDevice {
    fn open_any() -> Result<Self, RuntimeError> {
        let mut handle: hailort::hailo_device = std::ptr::null_mut();

        // SAFETY:
        // - A null device_id tells HailoRT to select an available device.
        // - handle points to writable storage for the returned device handle.
        let status = unsafe { hailort::hailo_create_device_by_id(std::ptr::null(), &mut handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_create_device_by_id",
                status,
            });
        }

        if handle.is_null() {
            return Err(RuntimeError::NullHandle {
                operation: "hailo_create_device_by_id",
            });
        }

        Ok(Self { handle })
    }

    fn configure(
        &self,
        hef: &HailoHef,
    ) -> Result<Vec<hailort::hailo_configured_network_group>, RuntimeError> {
        let capacity = hailort::HAILO_MAX_NETWORK_GROUPS as usize;

        let mut network_groups = vec![std::ptr::null_mut(); capacity];
        let mut count = network_groups.len();

        // SAFETY:
        // - self.handle is a valid, open device handle.
        // - hef.handle is a valid loaded HEF handle.
        // - A null params pointer requests HailoRT's default configuration.
        // - network_groups provides writable storage for `capacity` handles.
        // - count initially contains the array capacity and is writable.
        let status = unsafe {
            hailort::hailo_configure_device(
                self.handle,
                hef.handle,
                std::ptr::null_mut(),
                network_groups.as_mut_ptr(),
                &mut count,
            )
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_configure_device",
                status,
            });
        }

        if count > network_groups.len() {
            return Err(RuntimeError::CountExceedsCapacity {
                resource: "configured network groups",
                count,
                capacity: network_groups.len(),
            });
        }

        network_groups.truncate(count);

        if network_groups.iter().any(|handle| handle.is_null()) {
            return Err(RuntimeError::NullHandle {
                operation: "hailo_configure_device",
            });
        }

        Ok(network_groups)
    }
}

// Owns a service-backed VDevice. Do not treat it as a physical device.
struct HailoVDevice {
    handle: hailort::hailo_vdevice,
}

impl HailoVDevice {
    fn open_shared(group_id: &str) -> Result<Self, RuntimeError> {
        let group = CString::new(group_id).map_err(|_| RuntimeError::GroupIdContainsNullByte)?;
        let mut params = hailort::hailo_vdevice_params_t::default();
        let status = unsafe { hailort::hailo_init_vdevice_params(&mut params) };
        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_init_vdevice_params",
                status,
            });
        }
        // HailoRT 4.23: ROUND_ROBIN = 1 in hailo_scheduling_algorithm_e.
        params.scheduling_algorithm = 1 as _;
        params.multi_process_service = true;
        params.group_id = group.as_ptr();
        let mut handle: hailort::hailo_vdevice = ptr::null_mut();
        // HailoRT consumes the group ID during this call.
        let status = unsafe { hailort::hailo_create_vdevice(&mut params, &mut handle) };
        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_create_vdevice",
                status,
            });
        }
        if handle.is_null() {
            return Err(RuntimeError::NullHandle {
                operation: "hailo_create_vdevice",
            });
        }
        Ok(Self { handle })
    }

    fn configure(
        &self,
        hef: &HailoHef,
    ) -> Result<Vec<hailort::hailo_configured_network_group>, RuntimeError> {
        let capacity = hailort::HAILO_MAX_NETWORK_GROUPS as usize;
        let mut network_groups = vec![ptr::null_mut(); capacity];
        let mut count = network_groups.len();
        let status = unsafe {
            hailort::hailo_configure_vdevice(
                self.handle,
                hef.handle,
                ptr::null_mut(),
                network_groups.as_mut_ptr(),
                &mut count,
            )
        };
        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_configure_vdevice",
                status,
            });
        }
        if count > network_groups.len() {
            return Err(RuntimeError::CountExceedsCapacity {
                resource: "configured network groups",
                count,
                capacity: network_groups.len(),
            });
        }
        network_groups.truncate(count);
        if network_groups.iter().any(|handle| handle.is_null()) {
            return Err(RuntimeError::NullHandle {
                operation: "hailo_configure_vdevice",
            });
        }
        Ok(network_groups)
    }
}

impl Drop for HailoVDevice {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let status = unsafe { hailort::hailo_release_vdevice(self.handle) };
            if status != hailort::hailo_status_HAILO_SUCCESS {
                eprintln!("hailo_release_vdevice failed during cleanup: status={status}");
            }
            self.handle = ptr::null_mut();
        }
    }
}

enum DeviceHandle {
    Physical(HailoDevice),
    Shared(HailoVDevice),
}

impl DeviceHandle {
    fn configure(
        &self,
        hef: &HailoHef,
    ) -> Result<Vec<hailort::hailo_configured_network_group>, RuntimeError> {
        match self {
            Self::Physical(device) => device.configure(hef),
            Self::Shared(device) => device.configure(hef),
        }
    }
}

struct HailoInputVStreams {
    handles: Vec<hailort::hailo_input_vstream>,
}

impl HailoInputVStreams {
    fn create(
        network_group: hailort::hailo_configured_network_group,
    ) -> Result<Self, RuntimeError> {
        let capacity = hailort::HAILO_MAX_STREAMS_COUNT as usize;

        let mut params = vec![hailort::hailo_input_vstream_params_by_name_t::default(); capacity];
        let mut count = params.len();

        // SAFETY:
        // - network_group is a valid configured network-group handle.
        // - params provides writable storage for `capacity` parameter entries.
        // - count initially contains the parameter-array capacity.
        let status = unsafe {
            hailort::hailo_make_input_vstream_params(
                network_group,
                false,
                hailort::hailo_format_type_t_HAILO_FORMAT_TYPE_AUTO,
                params.as_mut_ptr(),
                &mut count,
            )
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_make_input_vstream_params",
                status,
            });
        }

        if count > params.len() {
            return Err(RuntimeError::CountExceedsCapacity {
                resource: "input virtual streams",
                count,
                capacity: params.len(),
            });
        }

        params.truncate(count);

        let mut handles = vec![std::ptr::null_mut(); params.len()];

        // SAFETY:
        // - network_group remains valid.
        // - params contains `count` initialized parameter entries.
        // - handles contains writable storage for `count` handles.
        let status = unsafe {
            hailort::hailo_create_input_vstreams(
                network_group,
                params.as_ptr(),
                params.len(),
                handles.as_mut_ptr(),
            )
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_create_input_vstreams",
                status,
            });
        }

        if handles.iter().any(|handle| handle.is_null()) {
            // Release any successfully created handles before returning.
            unsafe {
                hailort::hailo_release_input_vstreams(handles.as_mut_ptr(), handles.len());
            }

            return Err(RuntimeError::NullHandle {
                operation: "hailo_create_input_vstreams",
            });
        }

        Ok(Self { handles })
    }

    fn frame_size(&self, index: usize) -> Result<usize, RuntimeError> {
        let handle = *self
            .handles
            .get(index)
            .ok_or(RuntimeError::InvalidVStreamIndex {
                direction: "input",
                index,
            })?;

        let mut frame_size = 0usize;

        // SAFETY: handle is a valid input-vstream handle owned by self.
        let status =
            unsafe { hailort::hailo_get_input_vstream_frame_size(handle, &mut frame_size) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_get_input_vstream_frame_size",
                status,
            });
        }

        Ok(frame_size)
    }

    fn write(&self, index: usize, buffer: &[u8]) -> Result<(), RuntimeError> {
        let handle = *self
            .handles
            .get(index)
            .ok_or(RuntimeError::InvalidVStreamIndex {
                direction: "input",
                index,
            })?;

        let expected_size = self.frame_size(index)?;

        if buffer.len() != expected_size {
            return Err(RuntimeError::BufferSizeMismatch {
                direction: "input",
                index,
                actual: buffer.len(),
                expected: expected_size,
            });
        }

        // SAFETY:
        // - handle is a valid input-vstream handle.
        // - buffer remains valid for the duration of this blocking call.
        // - buffer length equals the required input frame size.
        let status = unsafe {
            hailort::hailo_vstream_write_raw_buffer(handle, buffer.as_ptr().cast(), buffer.len())
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_vstream_write_raw_buffer",
                status,
            });
        }

        Ok(())
    }

    fn flush(&self, index: usize) -> Result<(), RuntimeError> {
        let handle = *self
            .handles
            .get(index)
            .ok_or(RuntimeError::InvalidVStreamIndex {
                direction: "input",
                index,
            })?;

        // SAFETY: handle is a valid input-vstream handle owned by self.
        let status = unsafe { hailort::hailo_flush_input_vstream(handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_flush_input_vstream",
                status,
            });
        }

        Ok(())
    }
}

impl Drop for HailoInputVStreams {
    fn drop(&mut self) {
        if self.handles.is_empty() {
            return;
        }

        // SAFETY: self owns all handles in this array.
        let status = unsafe {
            hailort::hailo_release_input_vstreams(self.handles.as_mut_ptr(), self.handles.len())
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            eprintln!(
                "hailo_release_input_vstreams failed during cleanup: \
                 status={status}"
            );
        }

        self.handles.clear();
    }
}

struct HailoOutputVStreams {
    handles: Vec<hailort::hailo_output_vstream>,
}

pub(super) struct HailoVStreams {
    // Fields drop in declaration order. Both are released before
    // configured_device because HailoVStreams is created afterward.
    output: HailoOutputVStreams,
    input: HailoInputVStreams,
}

impl HailoOutputVStreams {
    fn create(
        network_group: hailort::hailo_configured_network_group,
    ) -> Result<Self, RuntimeError> {
        let capacity = hailort::HAILO_MAX_STREAMS_COUNT as usize;

        let mut params = vec![hailort::hailo_output_vstream_params_by_name_t::default(); capacity];
        let mut count = params.len();

        // SAFETY:
        // - network_group is a valid configured network-group handle.
        // - params provides writable storage for `capacity` parameter entries.
        // - count initially contains the parameter-array capacity.
        let status = unsafe {
            hailort::hailo_make_output_vstream_params(
                network_group,
                false,
                hailort::hailo_format_type_t_HAILO_FORMAT_TYPE_AUTO,
                params.as_mut_ptr(),
                &mut count,
            )
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_make_output_vstream_params",
                status,
            });
        }

        if count > params.len() {
            return Err(RuntimeError::CountExceedsCapacity {
                resource: "output virtual streams",
                count,
                capacity: params.len(),
            });
        }

        params.truncate(count);

        let mut handles = vec![std::ptr::null_mut(); params.len()];

        // SAFETY:
        // - network_group remains valid.
        // - params contains `count` initialized parameter entries.
        // - handles contains writable storage for `count` handles.
        let status = unsafe {
            hailort::hailo_create_output_vstreams(
                network_group,
                params.as_ptr(),
                params.len(),
                handles.as_mut_ptr(),
            )
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_create_output_vstreams",
                status,
            });
        }

        if handles.iter().any(|handle| handle.is_null()) {
            unsafe {
                hailort::hailo_release_output_vstreams(handles.as_mut_ptr(), handles.len());
            }

            return Err(RuntimeError::NullHandle {
                operation: "hailo_create_output_vstreams",
            });
        }

        Ok(Self { handles })
    }

    fn frame_size(&self, index: usize) -> Result<usize, RuntimeError> {
        let handle = *self
            .handles
            .get(index)
            .ok_or(RuntimeError::InvalidVStreamIndex {
                direction: "output",
                index,
            })?;

        let mut frame_size = 0usize;

        // SAFETY: handle is a valid output-vstream handle owned by self.
        let status =
            unsafe { hailort::hailo_get_output_vstream_frame_size(handle, &mut frame_size) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_get_output_vstream_frame_size",
                status,
            });
        }

        Ok(frame_size)
    }

    fn read(&self, index: usize, buffer: &mut [u8]) -> Result<(), RuntimeError> {
        let handle = *self
            .handles
            .get(index)
            .ok_or(RuntimeError::InvalidVStreamIndex {
                direction: "output",
                index,
            })?;

        let expected_size = self.frame_size(index)?;

        if buffer.len() != expected_size {
            return Err(RuntimeError::BufferSizeMismatch {
                direction: "output",
                index,
                actual: buffer.len(),
                expected: expected_size,
            });
        }

        // SAFETY:
        // - handle is a valid output-vstream handle.
        // - buffer is writable and remains valid during this blocking call.
        // - buffer length equals the required output frame size.
        let status = unsafe {
            hailort::hailo_vstream_read_raw_buffer(handle, buffer.as_mut_ptr().cast(), buffer.len())
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_vstream_read_raw_buffer",
                status,
            });
        }

        Ok(())
    }
}

impl Drop for HailoOutputVStreams {
    fn drop(&mut self) {
        if self.handles.is_empty() {
            return;
        }

        // SAFETY: self owns all handles in this array.
        let status = unsafe {
            hailort::hailo_release_output_vstreams(self.handles.as_mut_ptr(), self.handles.len())
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            eprintln!(
                "hailo_release_output_vstreams failed during cleanup: \
                 status={status}"
            );
        }

        self.handles.clear();
    }
}

impl Drop for HailoDevice {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: This object owns the valid device handle.
        let status = unsafe { hailort::hailo_release_device(self.handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            eprintln!("hailo_release_device failed during cleanup: status={status}");
        }

        self.handle = std::ptr::null_mut();
    }
}

pub(super) struct HailoActivatedNetworkGroup {
    handle: hailort::hailo_activated_network_group,
}

impl HailoActivatedNetworkGroup {
    pub(super) fn activate(
        network_group: hailort::hailo_configured_network_group,
    ) -> Result<Self, RuntimeError> {
        let mut handle: hailort::hailo_activated_network_group = std::ptr::null_mut();

        // SAFETY:
        // - network_group is a valid configured network-group handle.
        // - A null activation_params pointer requests default parameters.
        // - handle points to writable storage for the activated handle.
        let status = unsafe {
            hailort::hailo_activate_network_group(network_group, std::ptr::null_mut(), &mut handle)
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(RuntimeError::HailoRtCallFailed {
                operation: "hailo_activate_network_group",
                status,
            });
        }

        if handle.is_null() {
            return Err(RuntimeError::NullHandle {
                operation: "hailo_activate_network_group",
            });
        }

        Ok(Self { handle })
    }
}

impl Drop for HailoActivatedNetworkGroup {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: This wrapper owns the activated-network-group handle.
        let status = unsafe { hailort::hailo_deactivate_network_group(self.handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            eprintln!(
                "hailo_deactivate_network_group failed during cleanup: \
                 status={status}"
            );
        }

        self.handle = std::ptr::null_mut();
    }
}

pub(super) struct HailoConfiguredDevice {
    // Retained so the device outlives every configured-network-group handle.
    _device: DeviceHandle,
    network_groups: Vec<hailort::hailo_configured_network_group>,
}

impl HailoConfiguredDevice {
    pub(super) fn first_network_group(
        &self,
    ) -> Result<hailort::hailo_configured_network_group, RuntimeError> {
        self.network_groups
            .first()
            .copied()
            .ok_or(RuntimeError::NoConfiguredNetworkGroups)
    }
}

impl HailoConfiguredDevice {
    pub(super) fn open_and_configure(
        hef: &HailoHef,
        multi_process_service: bool,
        group_id: &str,
    ) -> Result<Self, RuntimeError> {
        println!("Opening a Hailo device");

        let device = if multi_process_service {
            println!("Using HailoRT multi-process service (group {group_id})");
            DeviceHandle::Shared(HailoVDevice::open_shared(group_id)?)
        } else {
            DeviceHandle::Physical(HailoDevice::open_any()?)
        };

        println!("Hailo device opened successfully");
        println!("Configuring the Hailo device from the HEF");

        let network_groups = device.configure(hef)?;

        if network_groups.is_empty() {
            return Err(RuntimeError::NoConfiguredNetworkGroups);
        }

        println!("Configured network groups: {}", network_groups.len());

        for (index, handle) in network_groups.iter().enumerate() {
            println!("  [{index}] handle={handle:p}");
        }

        Ok(Self {
            _device: device,
            network_groups,
        })
    }
}

impl HailoVStreams {
    pub(super) fn create_and_describe(
        network_group: hailort::hailo_configured_network_group,
    ) -> Result<Self, RuntimeError> {
        println!("Creating input virtual streams");

        let input = HailoInputVStreams::create(network_group)?;

        println!("Input virtual streams created: {}", input.handles.len());

        for index in 0..input.handles.len() {
            let frame_size = input.frame_size(index)?;

            println!("  input[{index}] frame size={frame_size} bytes");
        }

        println!("Creating output virtual streams");

        let output = HailoOutputVStreams::create(network_group)?;

        println!("Output virtual streams created: {}", output.handles.len());

        for index in 0..output.handles.len() {
            let frame_size = output.frame_size(index)?;

            println!("  output[{index}] frame size={frame_size} bytes");
        }

        Ok(Self { output, input })
    }

    pub(super) fn input_count(&self) -> usize {
        self.input.handles.len()
    }

    pub(super) fn output_count(&self) -> usize {
        self.output.handles.len()
    }

    pub(super) fn input_frame_size(&self, index: usize) -> Result<usize, RuntimeError> {
        self.input.frame_size(index)
    }

    pub(super) fn output_frame_size(&self, index: usize) -> Result<usize, RuntimeError> {
        self.output.frame_size(index)
    }

    pub(super) fn write_input(&self, index: usize, buffer: &[u8]) -> Result<(), RuntimeError> {
        self.input.write(index, buffer)
    }

    pub(super) fn read_output(&self, index: usize, buffer: &mut [u8]) -> Result<(), RuntimeError> {
        self.output.read(index, buffer)
    }

    pub(super) fn flush_input(&self, index: usize) -> Result<(), RuntimeError> {
        self.input.flush(index)
    }
}
