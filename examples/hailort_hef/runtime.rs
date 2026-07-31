use std::ffi::CString;
use std::ptr;

use anyhow::{Context, Result};

use super::hailort;

pub(super) struct HailoHef {
    handle: hailort::hailo_hef,
}

impl HailoHef {
    pub(super) fn load(path: &str) -> Result<Self, String> {
        let c_path =
            CString::new(path).map_err(|_| format!("HEF path contains a null byte: {path}"))?;

        let mut handle: hailort::hailo_hef = ptr::null_mut();

        // SAFETY:
        // - c_path is a valid, null-terminated C string.
        // - &mut handle is writable storage for the returned HEF handle.
        // - HailoRT retains the loaded HEF, not the path pointer.
        let status = unsafe { hailort::hailo_create_hef_file(&mut handle, c_path.as_ptr()) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!(
                "hailo_create_hef_file failed for {path}: status={status}"
            ));
        }

        if handle.is_null() {
            return Err(format!(
                "HailoRT returned success but supplied a null HEF handle: {path}"
            ));
        }

        Ok(Self { handle })
    }

    pub(super) fn network_groups(
        &self,
    ) -> Result<Vec<hailort::hailo_network_group_info_t>, String> {
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
            return Err(format!(
                "hailo_get_network_groups_infos failed: status={status}"
            ));
        }

        if count > infos.len() {
            return Err(format!(
                "HailoRT reported {count} network groups, \
                 but the buffer capacity is {}",
                infos.len()
            ));
        }

        infos.truncate(count);
        Ok(infos)
    }

    pub(super) fn vstream_infos(
        &self,
        network_group_name: &str,
    ) -> Result<Vec<hailort::hailo_vstream_info_t>, String> {
        const CAPACITY: usize = 64;

        let name = CString::new(network_group_name)
            .map_err(|_| "network-group name contains a null byte".to_string())?;

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
            return Err(format!(
                "hailo_hef_get_all_vstream_infos failed: status={status}"
            ));
        }

        if count > infos.len() {
            return Err(format!(
                "HailoRT reported {count} virtual streams, \
                 but the buffer capacity is {}",
                infos.len()
            ));
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
    fn open_any() -> Result<Self, String> {
        let mut handle: hailort::hailo_device = std::ptr::null_mut();

        // SAFETY:
        // - A null device_id tells HailoRT to select an available device.
        // - handle points to writable storage for the returned device handle.
        let status = unsafe { hailort::hailo_create_device_by_id(std::ptr::null(), &mut handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!("hailo_create_device_by_id failed: status={status}"));
        }

        if handle.is_null() {
            return Err(
                "hailo_create_device_by_id succeeded but returned a null handle".to_string(),
            );
        }

        Ok(Self { handle })
    }

    fn configure(
        &self,
        hef: &HailoHef,
    ) -> Result<Vec<hailort::hailo_configured_network_group>, String> {
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
            return Err(format!("hailo_configure_device failed: status={status}"));
        }

        if count > network_groups.len() {
            return Err(format!(
                "HailoRT reported {count} configured network groups, \
                 but the buffer capacity is {}",
                network_groups.len()
            ));
        }

        network_groups.truncate(count);

        if network_groups.iter().any(|handle| handle.is_null()) {
            return Err("hailo_configure_device returned a null network-group handle".to_string());
        }

        Ok(network_groups)
    }
}

struct HailoInputVStreams {
    handles: Vec<hailort::hailo_input_vstream>,
}

impl HailoInputVStreams {
    fn create(network_group: hailort::hailo_configured_network_group) -> Result<Self, String> {
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
            return Err(format!(
                "hailo_make_input_vstream_params failed: status={status}"
            ));
        }

        if count > params.len() {
            return Err(format!(
                "HailoRT reported {count} input virtual streams, \
                 but the capacity is {}",
                params.len()
            ));
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
            return Err(format!(
                "hailo_create_input_vstreams failed: status={status}"
            ));
        }

        if handles.iter().any(|handle| handle.is_null()) {
            // Release any successfully created handles before returning.
            unsafe {
                hailort::hailo_release_input_vstreams(handles.as_mut_ptr(), handles.len());
            }

            return Err("hailo_create_input_vstreams returned a null handle".to_string());
        }

        Ok(Self { handles })
    }

    fn frame_size(&self, index: usize) -> Result<usize, String> {
        let handle = *self
            .handles
            .get(index)
            .ok_or_else(|| format!("invalid input vstream index {index}"))?;

        let mut frame_size = 0usize;

        // SAFETY: handle is a valid input-vstream handle owned by self.
        let status =
            unsafe { hailort::hailo_get_input_vstream_frame_size(handle, &mut frame_size) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!(
                "hailo_get_input_vstream_frame_size failed: status={status}"
            ));
        }

        Ok(frame_size)
    }

    fn write(&self, index: usize, buffer: &[u8]) -> Result<(), String> {
        let handle = *self
            .handles
            .get(index)
            .ok_or_else(|| format!("invalid input vstream index {index}"))?;

        let expected_size = self.frame_size(index)?;

        if buffer.len() != expected_size {
            return Err(format!(
                "input buffer is {} bytes; expected {expected_size} bytes",
                buffer.len()
            ));
        }

        // SAFETY:
        // - handle is a valid input-vstream handle.
        // - buffer remains valid for the duration of this blocking call.
        // - buffer length equals the required input frame size.
        let status = unsafe {
            hailort::hailo_vstream_write_raw_buffer(handle, buffer.as_ptr().cast(), buffer.len())
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!(
                "hailo_vstream_write_raw_buffer failed: status={status}"
            ));
        }

        Ok(())
    }

    fn flush(&self, index: usize) -> Result<(), String> {
        let handle = *self
            .handles
            .get(index)
            .ok_or_else(|| format!("invalid input vstream index {index}"))?;

        // SAFETY: handle is a valid input-vstream handle owned by self.
        let status = unsafe { hailort::hailo_flush_input_vstream(handle) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!("hailo_flush_input_vstream failed: status={status}"));
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
    fn create(network_group: hailort::hailo_configured_network_group) -> Result<Self, String> {
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
            return Err(format!(
                "hailo_make_output_vstream_params failed: status={status}"
            ));
        }

        if count > params.len() {
            return Err(format!(
                "HailoRT reported {count} output virtual streams, \
                 but the capacity is {}",
                params.len()
            ));
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
            return Err(format!(
                "hailo_create_output_vstreams failed: status={status}"
            ));
        }

        if handles.iter().any(|handle| handle.is_null()) {
            unsafe {
                hailort::hailo_release_output_vstreams(handles.as_mut_ptr(), handles.len());
            }

            return Err("hailo_create_output_vstreams returned a null handle".to_string());
        }

        Ok(Self { handles })
    }

    fn frame_size(&self, index: usize) -> Result<usize, String> {
        let handle = *self
            .handles
            .get(index)
            .ok_or_else(|| format!("invalid output vstream index {index}"))?;

        let mut frame_size = 0usize;

        // SAFETY: handle is a valid output-vstream handle owned by self.
        let status =
            unsafe { hailort::hailo_get_output_vstream_frame_size(handle, &mut frame_size) };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!(
                "hailo_get_output_vstream_frame_size failed: status={status}"
            ));
        }

        Ok(frame_size)
    }

    fn read(&self, index: usize, buffer: &mut [u8]) -> Result<(), String> {
        let handle = *self
            .handles
            .get(index)
            .ok_or_else(|| format!("invalid output vstream index {index}"))?;

        let expected_size = self.frame_size(index)?;

        if buffer.len() != expected_size {
            return Err(format!(
                "output buffer is {} bytes; expected {expected_size} bytes",
                buffer.len()
            ));
        }

        // SAFETY:
        // - handle is a valid output-vstream handle.
        // - buffer is writable and remains valid during this blocking call.
        // - buffer length equals the required output frame size.
        let status = unsafe {
            hailort::hailo_vstream_read_raw_buffer(handle, buffer.as_mut_ptr().cast(), buffer.len())
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!(
                "hailo_vstream_read_raw_buffer failed: status={status}"
            ));
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
    ) -> Result<Self, String> {
        let mut handle: hailort::hailo_activated_network_group = std::ptr::null_mut();

        // SAFETY:
        // - network_group is a valid configured network-group handle.
        // - A null activation_params pointer requests default parameters.
        // - handle points to writable storage for the activated handle.
        let status = unsafe {
            hailort::hailo_activate_network_group(network_group, std::ptr::null_mut(), &mut handle)
        };

        if status != hailort::hailo_status_HAILO_SUCCESS {
            return Err(format!(
                "hailo_activate_network_group failed: status={status}"
            ));
        }

        if handle.is_null() {
            return Err("hailo_activate_network_group succeeded but returned \
                 a null handle"
                .to_string());
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
    _device: HailoDevice,
    network_groups: Vec<hailort::hailo_configured_network_group>,
}

impl HailoConfiguredDevice {
    pub(super) fn first_network_group(&self) -> Result<hailort::hailo_configured_network_group> {
        self.network_groups
            .first()
            .copied()
            .context("no configured network groups were returned")
    }
}

impl HailoConfiguredDevice {
    pub(super) fn open_and_configure(hef: &HailoHef) -> Result<Self> {
        println!("Opening an available Hailo device");

        let device = HailoDevice::open_any()
            .map_err(anyhow::Error::msg)
            .context("failed to open an available Hailo device")?;

        println!("Hailo device opened successfully");
        println!("Configuring the Hailo device from the HEF");

        let network_groups = device
            .configure(hef)
            .map_err(anyhow::Error::msg)
            .context("failed to configure the Hailo device from the HEF")?;

        if network_groups.is_empty() {
            anyhow::bail!("no configured network groups were returned");
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
    ) -> Result<Self> {
        println!("Creating input virtual streams");

        let input = HailoInputVStreams::create(network_group)
            .map_err(anyhow::Error::msg)
            .context("failed to create input virtual streams")?;

        println!("Input virtual streams created: {}", input.handles.len());

        for index in 0..input.handles.len() {
            let frame_size = input
                .frame_size(index)
                .map_err(anyhow::Error::msg)
                .with_context(|| {
                    format!("failed to obtain frame size for input virtual stream {index}")
                })?;

            println!("  input[{index}] frame size={frame_size} bytes");
        }

        println!("Creating output virtual streams");

        let output = HailoOutputVStreams::create(network_group)
            .map_err(anyhow::Error::msg)
            .context("failed to create output virtual streams")?;

        println!("Output virtual streams created: {}", output.handles.len());

        for index in 0..output.handles.len() {
            let frame_size = output
                .frame_size(index)
                .map_err(anyhow::Error::msg)
                .with_context(|| {
                    format!("failed to obtain frame size for output virtual stream {index}")
                })?;

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

    pub(super) fn input_frame_size(&self, index: usize) -> Result<usize, String> {
        self.input.frame_size(index)
    }

    pub(super) fn output_frame_size(&self, index: usize) -> Result<usize, String> {
        self.output.frame_size(index)
    }

    pub(super) fn write_input(&self, index: usize, buffer: &[u8]) -> Result<(), String> {
        self.input.write(index, buffer)
    }

    pub(super) fn read_output(&self, index: usize, buffer: &mut [u8]) -> Result<(), String> {
        self.output.read(index, buffer)
    }

    pub(super) fn flush_input(&self, index: usize) -> Result<(), String> {
        self.input.flush(index)
    }
}
