use std::ffi::{CStr, CString};
use std::fs::File;
use std::path::Path;
use std::ptr;

use ab_glyph::FontRef;
use anyhow::{Context, Result, bail};
use clap::Parser;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;

const DEFAULT_HEF_PATH: &str = "/models/hefs/yolov8n_hailo8_v2.18.0.hef";
const DEFAULT_IMAGE_PATH: &str = "/models/images/culdesac-latest.jpg";
const DEFAULT_OUTPUT_PATH: &str = "output/annotated.jpg";
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.50;

const ANNOTATION_FONT: &[u8] = include_bytes!("../assets/annotation-font.ttf");

const ANNOTATION_COLOR: Rgb<u8> = Rgb([255, 0, 0]);
const ANNOTATION_BOX_THICKNESS: u32 = 3;
const ANNOTATION_LABEL_OFFSET: u32 = 28;
const ANNOTATION_LABEL_FONT_SIZE: f32 = 24.0;

const COCO_CLASS_NAMES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

fn class_name(class_id: usize) -> &'static str {
    COCO_CLASS_NAMES.get(class_id).copied().unwrap_or("unknown")
}

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod hailort {
    include!(concat!(env!("OUT_DIR"), "/hailort_bindings.rs"));
}

struct HailoHef {
    handle: hailort::hailo_hef,
}

impl HailoHef {
    fn load(path: &str) -> Result<Self, String> {
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

    fn network_groups(&self) -> Result<Vec<hailort::hailo_network_group_info_t>, String> {
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

    fn vstream_infos(
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

struct HailoVStreams {
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

fn network_group_name(info: &hailort::hailo_network_group_info_t) -> String {
    // SAFETY: HailoRT documents this fixed-size name field as a
    // null-terminated C string.
    unsafe { CStr::from_ptr(info.name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn vstream_name(info: &hailort::hailo_vstream_info_t) -> String {
    // SAFETY: HailoRT returns this field as a null-terminated C string.
    unsafe { CStr::from_ptr(info.name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn vstream_network_name(info: &hailort::hailo_vstream_info_t) -> String {
    // SAFETY: HailoRT returns this field as a null-terminated C string.
    unsafe { CStr::from_ptr(info.network_name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn direction_name(direction: hailort::hailo_stream_direction_t) -> &'static str {
    match direction {
        hailort::hailo_stream_direction_t_HAILO_H2D_STREAM => "input (H2D)",
        hailort::hailo_stream_direction_t_HAILO_D2H_STREAM => "output (D2H)",
        _ => "unknown",
    }
}

fn format_type_name(format_type: hailort::hailo_format_type_t) -> &'static str {
    match format_type {
        hailort::hailo_format_type_t_HAILO_FORMAT_TYPE_UINT8 => "UINT8",
        hailort::hailo_format_type_t_HAILO_FORMAT_TYPE_UINT16 => "UINT16",
        hailort::hailo_format_type_t_HAILO_FORMAT_TYPE_FLOAT32 => "FLOAT32",
        _ => "other",
    }
}

fn format_order_name(order: hailort::hailo_format_order_t) -> &'static str {
    match order {
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_NHWC => "NHWC",
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_NCHW => "NCHW",
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_FCR => "FCR",
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_F8CR => "F8CR",
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS => "HAILO_NMS",
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK => {
            "HAILO_NMS_WITH_BYTE_MASK"
        }
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP => "HAILO_NMS_ON_CHIP",
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS => "HAILO_NMS_BY_CLASS",
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE => "HAILO_NMS_BY_SCORE",
        _ => "other",
    }
}

fn is_nms_order(order: hailort::hailo_format_order_t) -> bool {
    matches!(
        order,
        hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS
            | hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_WITH_BYTE_MASK
            | hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_ON_CHIP
            | hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS
            | hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE
    )
}

struct HailoActivatedNetworkGroup {
    handle: hailort::hailo_activated_network_group,
}

impl HailoActivatedNetworkGroup {
    fn activate(network_group: hailort::hailo_configured_network_group) -> Result<Self, String> {
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

struct HailoConfiguredDevice {
    // Retained so the device outlives every configured-network-group handle.
    _device: HailoDevice,
    network_groups: Vec<hailort::hailo_configured_network_group>,
}

impl HailoConfiguredDevice {
    fn first_network_group(&self) -> Result<hailort::hailo_configured_network_group> {
        self.network_groups
            .first()
            .copied()
            .context("no configured network groups were returned")
    }
}

fn load_rgb_image(
    image_path: &str,
    model_width: u32,
    model_height: u32,
) -> Result<PreprocessedImage, String> {
    println!("Loading input image: {image_path}");

    let source = image::open(image_path)
        .map_err(|error| format!("failed to open {image_path}: {error}"))?
        .to_rgb8();
    let original_width = source.width();
    let original_height = source.height();

    println!("Original image dimensions: {original_width}x{original_height}");

    let scale = (model_width as f32 / original_width as f32)
        .min(model_height as f32 / original_height as f32);

    let resized_width = (original_width as f32 * scale).round() as u32;
    let resized_height = (original_height as f32 * scale).round() as u32;

    let resized = image::imageops::resize(
        &source,
        resized_width,
        resized_height,
        image::imageops::FilterType::Triangle,
    );

    let pad_x = (model_width - resized_width) / 2;
    let pad_y = (model_height - resized_height) / 2;

    // YOLO letterbox padding is conventionally gray.
    let mut canvas =
        image::RgbImage::from_pixel(model_width, model_height, image::Rgb([114, 114, 114]));

    image::imageops::overlay(&mut canvas, &resized, i64::from(pad_x), i64::from(pad_y));

    println!(
        "Letterboxed image: resized={}x{} padding=(x={}, y={}) \
         model={}x{}",
        resized_width, resized_height, pad_x, pad_y, model_width, model_height,
    );

    Ok(PreprocessedImage {
        pixels: canvas.into_raw(),
        original_width,
        original_height,
        resized_width,
        resized_height,
        pad_x,
        pad_y,
    })
}

#[derive(Debug)]
struct Detection {
    class_id: usize,
    confidence: f32,
    ymin: f32,
    xmin: f32,
    ymax: f32,
    xmax: f32,
}

fn decode_nms_by_class(
    buffer: &[u8],
    class_count: usize,
    max_boxes_per_class: usize,
) -> Result<Vec<Detection>, String> {
    const VALUES_PER_BOX: usize = 5;
    const BYTES_PER_VALUE: usize = std::mem::size_of::<f32>();

    let values_per_class = 1 + max_boxes_per_class * VALUES_PER_BOX;

    let expected_size = class_count * values_per_class * BYTES_PER_VALUE;

    if buffer.len() != expected_size {
        return Err(format!(
            "NMS buffer contains {} bytes; expected {expected_size}",
            buffer.len()
        ));
    }

    fn read_f32(buffer: &[u8], value_index: usize) -> f32 {
        let byte_index = value_index * std::mem::size_of::<f32>();

        f32::from_le_bytes([
            buffer[byte_index],
            buffer[byte_index + 1],
            buffer[byte_index + 2],
            buffer[byte_index + 3],
        ])
    }

    let mut detections = Vec::new();

    for class_id in 0..class_count {
        let class_start = class_id * values_per_class;
        let count_value = read_f32(buffer, class_start);

        if !count_value.is_finite() || count_value < 0.0 || count_value.fract() != 0.0 {
            return Err(format!(
                "class {class_id} has invalid detection count {count_value}"
            ));
        }

        let detection_count = count_value as usize;

        if detection_count > max_boxes_per_class {
            return Err(format!(
                "class {class_id} reports {detection_count} detections; \
                 maximum is {max_boxes_per_class}"
            ));
        }

        for detection_index in 0..detection_count {
            let box_start = class_start + 1 + detection_index * VALUES_PER_BOX;

            let ymin = read_f32(buffer, box_start);
            let xmin = read_f32(buffer, box_start + 1);
            let ymax = read_f32(buffer, box_start + 2);
            let xmax = read_f32(buffer, box_start + 3);
            let confidence = read_f32(buffer, box_start + 4);

            if ![ymin, xmin, ymax, xmax, confidence]
                .iter()
                .all(|value| value.is_finite())
            {
                return Err(format!(
                    "class {class_id}, detection {detection_index} \
                     contains a non-finite value"
                ));
            }

            detections.push(Detection {
                class_id,
                confidence,
                ymin,
                xmin,
                ymax,
                xmax,
            });
        }
    }

    Ok(detections)
}

struct PreprocessedImage {
    pixels: Vec<u8>,
    original_width: u32,
    original_height: u32,
    resized_width: u32,
    resized_height: u32,
    pad_x: u32,
    pad_y: u32,
}

struct ModelMetadata {
    input_width: u32,
    input_height: u32,
    input_channels: u32,
    class_count: usize,
    max_detections_per_class: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct InferenceContract {
    input_frame_bytes: usize,
    output_frame_bytes: usize,
}

struct LoadedModel {
    hef: HailoHef,
    metadata: ModelMetadata,
}

#[derive(Debug)]
struct PixelBoundingBox {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

fn is_valid_bounding_box(bounding_box: &PixelBoundingBox) -> bool {
    bounding_box.left < bounding_box.right && bounding_box.top < bounding_box.bottom
}

fn detection_label_y(pixel_box: &PixelBoundingBox) -> u32 {
    pixel_box.top.saturating_sub(ANNOTATION_LABEL_OFFSET)
}

fn map_detection_to_original(
    detection: &Detection,
    image: &PreprocessedImage,
    model_width: u32,
    model_height: u32,
) -> PixelBoundingBox {
    let resized_left = detection.xmin * model_width as f32 - image.pad_x as f32;
    let resized_top = detection.ymin * model_height as f32 - image.pad_y as f32;
    let resized_right = detection.xmax * model_width as f32 - image.pad_x as f32;
    let resized_bottom = detection.ymax * model_height as f32 - image.pad_y as f32;

    let scale_x = image.original_width as f32 / image.resized_width as f32;
    let scale_y = image.original_height as f32 / image.resized_height as f32;

    let map_x = |value: f32| {
        (value * scale_x)
            .clamp(0.0, image.original_width as f32)
            .round() as u32
    };

    let map_y = |value: f32| {
        (value * scale_y)
            .clamp(0.0, image.original_height as f32)
            .round() as u32
    };

    PixelBoundingBox {
        left: map_x(resized_left),
        top: map_y(resized_top),
        right: map_x(resized_right),
        bottom: map_y(resized_bottom),
    }
}

fn save_annotated_image(annotated: &RgbImage, output_path: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(output_path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create output directory {}: {error}",
                parent.display()
            )
        })?;
    }

    annotated
        .save(output_path)
        .map_err(|error| format!("failed to save {output_path}: {error}"))?;

    Ok(())
}

fn load_annotation_image(image_path: &str) -> Result<RgbImage, String> {
    image::open(image_path)
        .map_err(|error| format!("failed to open {image_path} for annotation: {error}"))
        .map(|image| image.to_rgb8())
}

fn load_annotation_font() -> Result<FontRef<'static>, String> {
    FontRef::try_from_slice(ANNOTATION_FONT)
        .map_err(|error| format!("failed to load annotation font: {error}"))
}

fn draw_detection_bounding_box(annotated: &mut RgbImage, pixel_box: &PixelBoundingBox) {
    let width = pixel_box.right.saturating_sub(pixel_box.left).max(1);
    let height = pixel_box.bottom.saturating_sub(pixel_box.top).max(1);

    // Draw a three-pixel red bounding box.
    for thickness in 0..ANNOTATION_BOX_THICKNESS {
        let left = pixel_box.left.saturating_add(thickness);
        let top = pixel_box.top.saturating_add(thickness);
        let box_width = width.saturating_sub(thickness * 2).max(1);
        let box_height = height.saturating_sub(thickness * 2).max(1);

        let rectangle = Rect::at(left as i32, top as i32).of_size(box_width, box_height);

        draw_hollow_rect_mut(annotated, rectangle, ANNOTATION_COLOR);
    }
}

fn format_detection_label(detection: &Detection) -> String {
    format!(
        "{} {:.1}%",
        class_name(detection.class_id),
        detection.confidence * 100.0,
    )
}

fn draw_detection_label(
    annotated: &mut RgbImage,
    font: &FontRef<'_>,
    detection: &Detection,
    pixel_box: &PixelBoundingBox,
) {
    let label = format_detection_label(detection);
    let label_y = detection_label_y(pixel_box);

    draw_text_mut(
        annotated,
        ANNOTATION_COLOR,
        pixel_box.left as i32,
        label_y as i32,
        ANNOTATION_LABEL_FONT_SIZE,
        font,
        &label,
    );
}

fn draw_detection_annotation(
    annotated: &mut RgbImage,
    font: &FontRef<'_>,
    detection: &Detection,
    pixel_box: &PixelBoundingBox,
) {
    draw_detection_bounding_box(annotated, pixel_box);
    draw_detection_label(annotated, font, detection, pixel_box);
}

fn draw_detection_annotations(
    annotated: &mut RgbImage,
    font: &FontRef<'_>,
    detections: &[Detection],
    preprocessed: &PreprocessedImage,
    model_width: u32,
    model_height: u32,
) {
    for detection in detections {
        let pixel_box =
            map_detection_to_original(detection, preprocessed, model_width, model_height);

        if !is_valid_bounding_box(&pixel_box) {
            continue;
        }

        draw_detection_annotation(annotated, font, detection, &pixel_box);
    }
}

fn annotate_image(
    image_path: &str,
    output_path: &str,
    detections: &[Detection],
    preprocessed: &PreprocessedImage,
    model_width: u32,
    model_height: u32,
) -> Result<(), String> {
    let mut annotated = load_annotation_image(image_path)?;

    let font = load_annotation_font()?;

    draw_detection_annotations(
        &mut annotated,
        &font,
        detections,
        preprocessed,
        model_width,
        model_height,
    );

    save_annotated_image(&annotated, output_path)?;

    Ok(())
}

fn parse_confidence_threshold(value: &str) -> Result<f32, String> {
    let threshold = value
        .parse::<f32>()
        .map_err(|error| format!("invalid confidence threshold '{value}': {error}"))?;

    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!(
            "confidence threshold must be between 0.0 and 1.0; received {value}"
        ));
    }

    Ok(threshold)
}

#[derive(Debug, Parser)]
#[command(
    name = "hailort-hef",
    version,
    about = "Run YOLOv8 inference on a Hailo-8 device and annotate an image"
)]
struct Cli {
    /// Path to the Hailo executable format model
    #[arg(long, value_name = "FILE", default_value = DEFAULT_HEF_PATH)]
    hef: String,

    /// Path to the input image
    #[arg(short, long, value_name = "FILE", default_value = DEFAULT_IMAGE_PATH)]
    input: String,

    /// Path for the annotated output image
    #[arg(short, long, value_name = "FILE", default_value = DEFAULT_OUTPUT_PATH)]
    output: String,

    /// Minimum detection confidence from 0.0 through 1.0
    #[arg(
        short,
        long,
        value_name = "NUMBER",
        default_value_t = DEFAULT_CONFIDENCE_THRESHOLD,
        value_parser = parse_confidence_threshold
    )]
    confidence: f32,
}

fn validate_readable_file(path: &str, description: &str) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("{description} '{path}' cannot be accessed"))?;

    if !metadata.is_file() {
        bail!("{description} '{path}' is not a regular file");
    }

    File::open(path)
        .with_context(|| format!("{description} '{path}' cannot be opened for reading"))?;

    Ok(())
}

fn derive_model_metadata(vstreams: &[hailort::hailo_vstream_info_t]) -> Result<ModelMetadata> {
    let input = vstreams
        .iter()
        .find(|vstream| {
            vstream.direction == hailort::hailo_stream_direction_t_HAILO_H2D_STREAM
                && !is_nms_order(vstream.format.order)
        })
        .context("HEF contains no non-NMS input virtual stream")?;

    let output = vstreams
        .iter()
        .find(|vstream| {
            vstream.direction == hailort::hailo_stream_direction_t_HAILO_D2H_STREAM
                && vstream.format.order
                    == hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS
        })
        .context("HEF contains no HAILO_NMS_BY_CLASS output virtual stream")?;

    // SAFETY: The selected input stream has a non-NMS format order,
    // so the shape union member is active.
    let input_shape = unsafe { input.__bindgen_anon_1.shape };

    // SAFETY: The selected output stream uses HAILO_NMS_BY_CLASS,
    // so the nms_shape union member is active.
    let nms_shape = unsafe { output.__bindgen_anon_1.nms_shape };

    let metadata = ModelMetadata {
        input_width: input_shape.width,
        input_height: input_shape.height,
        input_channels: input_shape.features,
        class_count: nms_shape.number_of_classes as usize,
        max_detections_per_class: nms_shape.max_bboxes_per_class as usize,
    };

    if metadata.input_width == 0 || metadata.input_height == 0 || metadata.input_channels == 0 {
        bail!(
            "HEF reports an invalid input shape: {}x{}x{}",
            metadata.input_height,
            metadata.input_width,
            metadata.input_channels
        );
    }

    if metadata.input_channels != 3 {
        bail!(
            "RGB preprocessing requires 3 input channels, but the HEF reports {}",
            metadata.input_channels
        );
    }

    if metadata.class_count == 0 {
        bail!("HEF reports zero NMS classes");
    }

    if metadata.max_detections_per_class == 0 {
        bail!("HEF reports zero maximum detections per class");
    }

    println!(
        "Derived model metadata: input={}x{}x{} \
         classes={} max_detections_per_class={}",
        metadata.input_width,
        metadata.input_height,
        metadata.input_channels,
        metadata.class_count,
        metadata.max_detections_per_class
    );

    Ok(metadata)
}

fn describe_vstream(index: usize, vstream: &hailort::hailo_vstream_info_t) {
    println!(
        "      [{index}] name={} network={}",
        vstream_name(vstream),
        vstream_network_name(vstream)
    );

    println!(
        "          direction={} format={} order={} flags={}",
        direction_name(vstream.direction),
        format_type_name(vstream.format.type_),
        format_order_name(vstream.format.order),
        vstream.format.flags
    );

    println!(
        "          raw format: type={} order={}",
        vstream.format.type_, vstream.format.order
    );

    if is_nms_order(vstream.format.order) {
        // SAFETY: For an NMS format order, nms_shape is active.
        let nms = unsafe { vstream.__bindgen_anon_1.nms_shape };

        println!(
            "          NMS: classes={} max_per_class={} \
             max_total={} max_mask_bytes={}",
            nms.number_of_classes,
            nms.max_bboxes_per_class,
            nms.max_bboxes_total,
            nms.max_accumulated_mask_size
        );

        if vstream.format.order
            == hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS
        {
            let count_bytes = std::mem::size_of::<f32>();
            let bbox_bytes = 5 * std::mem::size_of::<f32>();

            let max_frame_bytes = nms.number_of_classes as usize
                * (count_bytes + nms.max_bboxes_per_class as usize * bbox_bytes);

            println!(
                "          maximum output frame size={} bytes",
                max_frame_bytes
            );
        }
    } else {
        // SAFETY: For non-NMS format orders, shape is active.
        let shape = unsafe { vstream.__bindgen_anon_1.shape };

        println!(
            "          shape={}x{}x{}",
            shape.height, shape.width, shape.features
        );
    }

    println!(
        "          quantization: zero_point={} scale={} range=[{}, {}]",
        vstream.quant_info.qp_zp,
        vstream.quant_info.qp_scale,
        vstream.quant_info.limvals_min,
        vstream.quant_info.limvals_max
    );
}

fn describe_model(
    network_groups: &[hailort::hailo_network_group_info_t],
    selected_group_name: &str,
    vstreams: &[hailort::hailo_vstream_info_t],
) {
    println!("Network groups: {}", network_groups.len());

    for (index, info) in network_groups.iter().enumerate() {
        println!(
            "  [{index}] name={} multi_context={}",
            network_group_name(info),
            info.is_multi_context
        );
    }

    println!("    Selected network group: {selected_group_name}");
    println!("    Virtual streams: {}", vstreams.len());

    for (index, vstream) in vstreams.iter().enumerate() {
        describe_vstream(index, vstream);
    }
}

impl LoadedModel {
    fn load_and_describe(hef_path: &str) -> Result<Self> {
        println!("Loading HEF: {hef_path}");

        let hef = HailoHef::load(hef_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load HEF '{hef_path}'"))?;

        println!("HEF loaded successfully");

        let network_groups = hef
            .network_groups()
            .map_err(anyhow::Error::msg)
            .context("failed to obtain HEF network-group information")?;

        let first_network_group = network_groups
            .first()
            .context("HEF contains no network groups")?;

        let first_group_name = network_group_name(first_network_group);

        let vstreams = hef
            .vstream_infos(&first_group_name)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "failed to obtain virtual-stream information for \
                     network group '{first_group_name}'"
                )
            })?;

        describe_model(&network_groups, &first_group_name, &vstreams);

        let metadata = derive_model_metadata(&vstreams)?;

        Ok(Self { hef, metadata })
    }
}

fn open_and_configure_device(hef: &HailoHef) -> Result<HailoConfiguredDevice> {
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

    Ok(HailoConfiguredDevice {
        _device: device,
        network_groups,
    })
}

fn create_vstreams(
    network_group: hailort::hailo_configured_network_group,
) -> Result<HailoVStreams> {
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

    Ok(HailoVStreams { output, input })
}

fn derive_inference_contract(metadata: &ModelMetadata) -> Result<InferenceContract> {
    if metadata.input_width == 0 || metadata.input_height == 0 || metadata.input_channels == 0 {
        bail!(
            "cannot derive an inference contract from input shape {}x{}x{}",
            metadata.input_width,
            metadata.input_height,
            metadata.input_channels
        );
    }

    if metadata.class_count == 0 {
        bail!("cannot derive an inference contract with zero classes");
    }

    if metadata.max_detections_per_class == 0 {
        bail!("cannot derive an inference contract with zero maximum detections per class");
    }

    let input_width = usize::try_from(metadata.input_width)
        .context("input width cannot be represented as usize")?;

    let input_height = usize::try_from(metadata.input_height)
        .context("input height cannot be represented as usize")?;

    let input_channels = usize::try_from(metadata.input_channels)
        .context("input channel count cannot be represented as usize")?;

    let input_frame_bytes = input_width
        .checked_mul(input_height)
        .and_then(|size| size.checked_mul(input_channels))
        .context("input frame-size calculation overflowed")?;

    const VALUES_PER_DETECTION: usize = 5;
    const BYTES_PER_F32: usize = std::mem::size_of::<f32>();

    let values_per_class = metadata
        .max_detections_per_class
        .checked_mul(VALUES_PER_DETECTION)
        .and_then(|value| value.checked_add(1))
        .context("NMS values-per-class calculation overflowed")?;

    let output_frame_bytes = metadata
        .class_count
        .checked_mul(values_per_class)
        .and_then(|size| size.checked_mul(BYTES_PER_F32))
        .context("NMS output frame-size calculation overflowed")?;

    Ok(InferenceContract {
        input_frame_bytes,
        output_frame_bytes,
    })
}

fn validate_inference_contract(metadata: &ModelMetadata, vstreams: &HailoVStreams) -> Result<()> {
    if vstreams.input.handles.len() != 1 {
        bail!(
            "inference pipeline requires exactly one input virtual stream; HEF created {}",
            vstreams.input.handles.len()
        );
    }

    if vstreams.output.handles.len() != 1 {
        bail!(
            "inference pipeline requires exactly one output virtual stream; HEF created {}",
            vstreams.output.handles.len()
        );
    }

    let contract = derive_inference_contract(metadata)?;

    let actual_input_size = vstreams
        .input
        .frame_size(0)
        .map_err(anyhow::Error::msg)
        .context("failed to obtain runtime input frame size")?;

    if actual_input_size != contract.input_frame_bytes {
        bail!(
            "input contract mismatch: metadata requires {} bytes \
             ({}x{}x{} UINT8), but the input virtual stream requires {} bytes",
            contract.input_frame_bytes,
            metadata.input_width,
            metadata.input_height,
            metadata.input_channels,
            actual_input_size
        );
    }

    let actual_output_size = vstreams
        .output
        .frame_size(0)
        .map_err(anyhow::Error::msg)
        .context("failed to obtain runtime output frame size")?;

    if actual_output_size != contract.output_frame_bytes {
        bail!(
            "output contract mismatch: metadata requires {} bytes \
             ({} classes, {} maximum detections per class), \
             but the output virtual stream requires {} bytes",
            contract.output_frame_bytes,
            metadata.class_count,
            metadata.max_detections_per_class,
            actual_output_size
        );
    }

    println!(
        "Inference contract validated: input={} bytes output={} bytes",
        actual_input_size, actual_output_size
    );

    Ok(())
}

fn activate_network_group(
    network_group: hailort::hailo_configured_network_group,
) -> Result<HailoActivatedNetworkGroup> {
    println!("Activating the configured network group");

    let activated = HailoActivatedNetworkGroup::activate(network_group)
        .map_err(anyhow::Error::msg)
        .context("failed to activate the configured network group")?;

    println!("Network group activated successfully");

    Ok(activated)
}

fn infer_one(vstreams: &HailoVStreams, input_frame: &[u8]) -> Result<Vec<u8>> {
    let input_frame_size = vstreams
        .input
        .frame_size(0)
        .map_err(anyhow::Error::msg)
        .context("failed to obtain the input frame size")?;

    if input_frame.len() != input_frame_size {
        anyhow::bail!(
            "input frame contains {} bytes; HailoRT expects {} bytes",
            input_frame.len(),
            input_frame_size
        );
    }

    let output_frame_size = vstreams
        .output
        .frame_size(0)
        .map_err(anyhow::Error::msg)
        .context("failed to obtain the output frame size")?;

    let mut output_frame = vec![0_u8; output_frame_size];

    println!(
        "Writing one preprocessed RGB image: {} bytes",
        input_frame.len()
    );

    vstreams
        .input
        .write(0, input_frame)
        .map_err(anyhow::Error::msg)
        .context("failed to write the input frame")?;

    println!("Input frame written successfully");
    println!("Reading one inference result");

    vstreams
        .output
        .read(0, &mut output_frame)
        .map_err(anyhow::Error::msg)
        .context("failed to read the output frame")?;

    println!(
        "Output frame read successfully: {} bytes",
        output_frame.len()
    );

    vstreams
        .input
        .flush(0)
        .map_err(anyhow::Error::msg)
        .context("failed to flush the input virtual stream")?;

    println!("Input virtual stream flushed successfully");

    Ok(output_frame)
}

fn postprocess_yolov8(
    output_frame: &[u8],
    confidence_threshold: f32,
    class_count: usize,
    max_detections_per_class: usize,
) -> Result<Vec<Detection>> {
    let mut detections = decode_nms_by_class(output_frame, class_count, max_detections_per_class)
        .map_err(anyhow::Error::msg)
        .context("failed to decode YOLOv8 NMS output")?;

    let unfiltered_count = detections.len();

    detections.retain(|detection| detection.confidence >= confidence_threshold);

    println!(
        "Decoded detections: {unfiltered_count}; \
         retained at threshold {confidence_threshold:.2}: {}",
        detections.len()
    );

    Ok(detections)
}

fn report_and_annotate_detections(
    image_path: &str,
    output_path: &str,
    detections: &[Detection],
    preprocessed_image: &PreprocessedImage,
    model_width: u32,
    model_height: u32,
) -> Result<()> {
    for (index, detection) in detections.iter().enumerate() {
        let pixel_box =
            map_detection_to_original(detection, preprocessed_image, model_width, model_height);

        println!(
            "  [{index}] class={} name={} confidence={:.3} \
             normalized=(ymin={:.4}, xmin={:.4}, ymax={:.4}, xmax={:.4}) \
             original=(left={}, top={}, right={}, bottom={})",
            detection.class_id,
            class_name(detection.class_id),
            detection.confidence,
            detection.ymin,
            detection.xmin,
            detection.ymax,
            detection.xmax,
            pixel_box.left,
            pixel_box.top,
            pixel_box.right,
            pixel_box.bottom,
        );
    }

    annotate_image(
        image_path,
        output_path,
        detections,
        preprocessed_image,
        model_width,
        model_height,
    )
    .map_err(anyhow::Error::msg)
    .context("failed to annotate the input image")?;

    println!("Annotated image saved: {output_path}");

    Ok(())
}

fn process_image(
    vstreams: &HailoVStreams,
    metadata: &ModelMetadata,
    image_path: &str,
    output_path: &str,
    confidence_threshold: f32,
) -> Result<()> {
    let preprocessed_image =
        load_rgb_image(image_path, metadata.input_width, metadata.input_height)
            .map_err(anyhow::Error::msg)
            .context("failed to load and preprocess the input image")?;

    let output_frame = infer_one(vstreams, &preprocessed_image.pixels)?;

    let detections = postprocess_yolov8(
        &output_frame,
        confidence_threshold,
        metadata.class_count,
        metadata.max_detections_per_class,
    )?;

    report_and_annotate_detections(
        image_path,
        output_path,
        &detections,
        &preprocessed_image,
        metadata.input_width,
        metadata.input_height,
    )
}

fn run_inference(
    hef_path: &str,
    image_path: &str,
    output_path: &str,
    confidence_threshold: f32,
) -> Result<()> {
    println!("Confidence threshold: {confidence_threshold:.2}");

    let model = LoadedModel::load_and_describe(hef_path)?;

    let configured_device = open_and_configure_device(&model.hef)?;
    let network_group = configured_device.first_network_group()?;

    let vstreams = create_vstreams(network_group)?;
    validate_inference_contract(&model.metadata, &vstreams)?;
    let activated_network_group = activate_network_group(network_group)?;

    process_image(
        &vstreams,
        &model.metadata,
        image_path,
        output_path,
        confidence_threshold,
    )?;
    drop(activated_network_group);

    println!("Network group deactivated successfully");
    println!(
        "The virtual streams, configured network group, device, and HEF \
         will now be released automatically"
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let hef_path = cli.hef;
    let image_path = cli.input;
    let output_path = cli.output;
    let confidence_threshold = cli.confidence;

    validate_readable_file(&hef_path, "HEF file")?;
    validate_readable_file(&image_path, "input image")?;

    run_inference(&hef_path, &image_path, &output_path, confidence_threshold)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_metadata() -> ModelMetadata {
        ModelMetadata {
            input_width: 640,
            input_height: 640,
            input_channels: 3,
            class_count: 80,
            max_detections_per_class: 100,
        }
    }

    fn encode_f32_values(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn preprocessed_image(
        original_width: u32,
        original_height: u32,
        resized_width: u32,
        resized_height: u32,
        pad_x: u32,
        pad_y: u32,
    ) -> PreprocessedImage {
        PreprocessedImage {
            pixels: Vec::new(),
            original_width,
            original_height,
            resized_width,
            resized_height,
            pad_x,
            pad_y,
        }
    }

    fn valid_detection() -> Detection {
        Detection {
            class_id: 0,
            confidence: 0.95,
            ymin: 0.25,
            xmin: 0.25,
            ymax: 0.75,
            xmax: 0.75,
        }
    }

    #[test]
    fn derives_yolov8_inference_contract() {
        let contract =
            derive_inference_contract(&valid_metadata()).expect("valid metadata should succeed");

        assert_eq!(
            contract,
            InferenceContract {
                input_frame_bytes: 1_228_800,
                output_frame_bytes: 160_320,
            }
        );
    }

    #[test]
    fn rejects_zero_input_dimension() {
        let mut metadata = valid_metadata();
        metadata.input_width = 0;

        let error =
            derive_inference_contract(&metadata).expect_err("zero input width should be rejected");

        assert!(
            error
                .to_string()
                .contains("cannot derive an inference contract from input shape")
        );
    }

    #[test]
    fn rejects_zero_class_count() {
        let mut metadata = valid_metadata();
        metadata.class_count = 0;

        let error =
            derive_inference_contract(&metadata).expect_err("zero classes should be rejected");

        assert!(
            error
                .to_string()
                .contains("cannot derive an inference contract with zero classes")
        );
    }

    #[test]
    fn rejects_zero_maximum_detections() {
        let mut metadata = valid_metadata();
        metadata.max_detections_per_class = 0;

        let error = derive_inference_contract(&metadata)
            .expect_err("zero maximum detections should be rejected");

        assert!(
            error
                .to_string()
                .contains("zero maximum detections per class")
        );
    }

    #[test]
    fn detects_nms_values_per_class_overflow() {
        let mut metadata = valid_metadata();
        metadata.max_detections_per_class = usize::MAX;

        let error = derive_inference_contract(&metadata)
            .expect_err("NMS values-per-class arithmetic should overflow");

        assert!(
            error
                .to_string()
                .contains("NMS values-per-class calculation overflowed")
        );
    }

    #[test]
    fn detects_nms_output_size_overflow() {
        let mut metadata = valid_metadata();
        metadata.class_count = usize::MAX;

        let error = derive_inference_contract(&metadata)
            .expect_err("NMS output frame-size arithmetic should overflow");

        assert!(
            error
                .to_string()
                .contains("NMS output frame-size calculation overflowed")
        );
    }

    #[test]
    fn decodes_one_nms_detection() {
        // One class, with space for one detection:
        // [count, ymin, xmin, ymax, xmax, confidence]
        let buffer = encode_f32_values(&[1.0, 0.10, 0.20, 0.70, 0.80, 0.95]);

        let detections =
            decode_nms_by_class(&buffer, 1, 1).expect("valid NMS output should decode");

        assert_eq!(detections.len(), 1);

        let detection = &detections[0];

        assert_eq!(detection.class_id, 0);
        assert_eq!(detection.ymin, 0.10);
        assert_eq!(detection.xmin, 0.20);
        assert_eq!(detection.ymax, 0.70);
        assert_eq!(detection.xmax, 0.80);
        assert_eq!(detection.confidence, 0.95);
    }

    #[test]
    fn decodes_detections_from_multiple_classes() {
        // Two classes, with two reserved boxes per class.
        let buffer = encode_f32_values(&[
            // Class 0: one detection and one unused slot.
            1.0, 0.10, 0.20, 0.30, 0.40, 0.90, 0.0, 0.0, 0.0, 0.0, 0.0,
            // Class 1: two detections.
            2.0, 0.15, 0.25, 0.35, 0.45, 0.80, 0.50, 0.60, 0.70, 0.90, 0.75,
        ]);

        let detections =
            decode_nms_by_class(&buffer, 2, 2).expect("valid multi-class output should decode");

        assert_eq!(detections.len(), 3);

        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[0].confidence, 0.90);

        assert_eq!(detections[1].class_id, 1);
        assert_eq!(detections[1].confidence, 0.80);

        assert_eq!(detections[2].class_id, 1);
        assert_eq!(detections[2].confidence, 0.75);
    }

    #[test]
    fn decodes_zero_detections() {
        // Two classes, with one reserved box per class.
        let buffer =
            encode_f32_values(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let detections =
            decode_nms_by_class(&buffer, 2, 1).expect("empty NMS output should be valid");

        assert!(detections.is_empty());
    }

    #[test]
    fn rejects_truncated_nms_output() {
        let buffer = encode_f32_values(&[
            1.0, 0.10, 0.20, 0.70, 0.80,
            // Missing confidence.
        ]);

        let error = decode_nms_by_class(&buffer, 1, 1)
            .expect_err("truncated NMS output should be rejected");

        assert!(error.contains("NMS buffer contains"));
        assert!(error.contains("expected 24"));
    }

    #[test]
    fn rejects_detection_count_above_maximum() {
        let buffer = encode_f32_values(&[2.0, 0.10, 0.20, 0.30, 0.40, 0.90]);

        let error = decode_nms_by_class(&buffer, 1, 1)
            .expect_err("excessive detection count should be rejected");

        assert!(error.contains("class 0 reports 2 detections"));
        assert!(error.contains("maximum is 1"));
    }

    #[test]
    fn rejects_non_finite_detection_value() {
        let buffer = encode_f32_values(&[1.0, 0.10, 0.20, f32::NAN, 0.80, 0.95]);

        let error = decode_nms_by_class(&buffer, 1, 1)
            .expect_err("non-finite detection data should be rejected");

        assert!(error.contains("class 0, detection 0"));
        assert!(error.contains("contains a non-finite value"));
    }

    #[test]
    fn maps_snowflake_detection_to_original_image() {
        let image = preprocessed_image(
            257, 367, // Original image
            448, 640, // Letterboxed resize
            96, 0, // Horizontal padding
        );

        let detection = Detection {
            class_id: 0,
            confidence: 0.862,
            ymin: 0.2513,
            xmin: 0.1482,
            ymax: 0.9996,
            xmax: 0.8460,
        };

        let bounding_box = map_detection_to_original(&detection, &image, 640, 640);

        assert_eq!(bounding_box.left, 0);
        assert_eq!(bounding_box.top, 92);
        assert_eq!(bounding_box.right, 256);
        assert_eq!(bounding_box.bottom, 367);

        // Coordinate mapping must not alter detection metadata.
        assert_eq!(detection.class_id, 0);
        assert_eq!(detection.confidence, 0.862);
    }

    #[test]
    fn removes_horizontal_letterbox_padding() {
        // A 100x200 portrait image scales to 320x640 and receives
        // 160 pixels of padding on both horizontal sides.
        let image = preprocessed_image(100, 200, 320, 640, 160, 0);
        let detection = valid_detection();

        let bounding_box = map_detection_to_original(&detection, &image, 640, 640);

        assert_eq!(bounding_box.left, 0);
        assert_eq!(bounding_box.top, 50);
        assert_eq!(bounding_box.right, 100);
        assert_eq!(bounding_box.bottom, 150);
    }

    #[test]
    fn removes_vertical_letterbox_padding() {
        // A 200x100 landscape image scales to 640x320 and receives
        // 160 pixels of padding above and below.
        let image = preprocessed_image(200, 100, 640, 320, 0, 160);

        let detection = valid_detection();

        let bounding_box = map_detection_to_original(&detection, &image, 640, 640);

        assert_eq!(bounding_box.left, 50);
        assert_eq!(bounding_box.top, 0);
        assert_eq!(bounding_box.right, 150);
        assert_eq!(bounding_box.bottom, 100);
    }

    #[test]
    fn clamps_detection_to_original_image_boundaries() {
        let image = preprocessed_image(640, 640, 640, 640, 0, 0);

        let detection = Detection {
            class_id: 0,
            confidence: 0.95,
            ymin: -0.10,
            xmin: -0.20,
            ymax: 1.10,
            xmax: 1.20,
        };

        let bounding_box = map_detection_to_original(&detection, &image, 640, 640);

        assert_eq!(bounding_box.left, 0);
        assert_eq!(bounding_box.top, 0);
        assert_eq!(bounding_box.right, 640);
        assert_eq!(bounding_box.bottom, 640);
    }

    #[test]
    fn filters_detections_at_confidence_threshold() {
        // Two classes with one reserved detection slot each.
        let buffer = encode_f32_values(&[
            // Class 0: below the threshold.
            1.0, 0.10, 0.20, 0.30, 0.40, 0.79, // Class 1: exactly at the threshold.
            1.0, 0.50, 0.60, 0.70, 0.80, 0.80,
        ]);

        let detections = postprocess_yolov8(&buffer, 0.80, 2, 1)
            .expect("valid NMS output should be postprocessed");

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_id, 1);
        assert_eq!(detections[0].confidence, 0.80);
    }

    #[test]
    fn accepts_valid_bounding_box() {
        let bounding_box = PixelBoundingBox {
            left: 10,
            top: 20,
            right: 110,
            bottom: 220,
        };

        assert!(is_valid_bounding_box(&bounding_box));
    }

    #[test]
    fn rejects_zero_width_bounding_box() {
        let bounding_box = PixelBoundingBox {
            left: 50,
            top: 20,
            right: 50,
            bottom: 220,
        };

        assert!(!is_valid_bounding_box(&bounding_box));
    }

    #[test]
    fn rejects_zero_height_bounding_box() {
        let bounding_box = PixelBoundingBox {
            left: 10,
            top: 75,
            right: 110,
            bottom: 75,
        };

        assert!(!is_valid_bounding_box(&bounding_box));
    }

    #[test]
    fn rejects_horizontally_inverted_bounding_box() {
        let bounding_box = PixelBoundingBox {
            left: 110,
            top: 20,
            right: 10,
            bottom: 220,
        };

        assert!(!is_valid_bounding_box(&bounding_box));
    }

    #[test]
    fn rejects_vertically_inverted_bounding_box() {
        let bounding_box = PixelBoundingBox {
            left: 10,
            top: 220,
            right: 110,
            bottom: 20,
        };

        assert!(!is_valid_bounding_box(&bounding_box));
    }

    #[test]
    fn skips_invalid_bounding_box_during_annotation() {
        let fixture = AnnotationTestFixture::new(32, 32);
        let preprocessed = preprocessed_image(32, 32, 32, 32, 0, 0);

        let invalid_detection = Detection {
            xmin: 0.75,
            xmax: 0.25,
            ..valid_detection()
        };

        annotate_image(
            fixture.input_path(),
            fixture.output_path(),
            &[invalid_detection],
            &preprocessed,
            32,
            32,
        )
        .expect("annotation should succeed while skipping the invalid box");

        assert_eq!(fixture.load_output(), fixture.original);
    }

    #[test]
    fn draws_valid_bounding_box_during_annotation() {
        let fixture = AnnotationTestFixture::new(32, 32);
        let preprocessed = preprocessed_image(32, 32, 32, 32, 0, 0);
        let valid_detection = valid_detection();

        annotate_image(
            fixture.input_path(),
            fixture.output_path(),
            &[valid_detection],
            &preprocessed,
            32,
            32,
        )
        .expect("valid detection should be annotated");

        let annotated = fixture.load_output();

        assert_ne!(annotated, fixture.original);

        // Normalized coordinate 0.25 maps to pixel 8 in a 32x32 image.
        assert_eq!(annotated.get_pixel(8, 8), &Rgb([255, 0, 0]));
    }

    struct AnnotationTestFixture {
        directory: std::path::PathBuf,
        input_path: std::path::PathBuf,
        output_path: std::path::PathBuf,
        original: RgbImage,
    }

    impl AnnotationTestFixture {
        fn new(width: u32, height: u32) -> Self {
            let unique_id = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after the Unix epoch")
                .as_nanos();

            let directory = std::env::temp_dir().join(format!(
                "hybrid-inference-annotation-{}-{unique_id}",
                std::process::id()
            ));

            std::fs::create_dir_all(&directory)
                .expect("temporary test directory should be created");

            let input_path = directory.join("input.png");
            let output_path = directory.join("output.png");
            let original = RgbImage::from_pixel(width, height, Rgb([114, 114, 114]));

            original
                .save(&input_path)
                .expect("temporary input image should be saved");

            Self {
                directory,
                input_path,
                output_path,
                original,
            }
        }

        fn input_path(&self) -> &str {
            self.input_path
                .to_str()
                .expect("input path should contain valid UTF-8")
        }

        fn output_path(&self) -> &str {
            self.output_path
                .to_str()
                .expect("output path should contain valid UTF-8")
        }

        fn load_output(&self) -> RgbImage {
            image::open(&self.output_path)
                .expect("annotated output image should open")
                .to_rgb8()
        }
    }

    impl Drop for AnnotationTestFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    #[test]
    fn formats_detection_label() {
        let detection = Detection {
            class_id: 0,
            confidence: 0.862,
            ..valid_detection()
        };

        assert_eq!(format_detection_label(&detection), "person 86.2%");
    }

    #[test]
    fn positions_detection_label_above_bounding_box() {
        let bounding_box = PixelBoundingBox {
            left: 10,
            top: 100,
            right: 110,
            bottom: 220,
        };

        assert_eq!(detection_label_y(&bounding_box), 72);
    }

    #[test]
    fn positions_detection_label_at_top_edge_without_underflow() {
        let bounding_box = PixelBoundingBox {
            left: 10,
            top: 20,
            right: 110,
            bottom: 220,
        };

        assert_eq!(detection_label_y(&bounding_box), 0);
    }

    #[test]
    fn formats_unknown_detection_class_label() {
        let detection = Detection {
            class_id: COCO_CLASS_NAMES.len(),
            confidence: 0.50,
            ..valid_detection()
        };

        assert_eq!(format_detection_label(&detection), "unknown 50.0%");
    }
}
