use anyhow::{Context, Result, bail};
use std::ffi::CStr;

use super::hailort;
use super::runtime::HailoHef;
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

pub(super) struct ModelMetadata {
    pub(super) input_width: u32,
    pub(super) input_height: u32,
    pub(super) input_channels: u32,
    pub(super) class_count: usize,
    pub(super) max_detections_per_class: usize,
}

pub(super) struct LoadedModel {
    pub(super) hef: HailoHef,
    pub(super) metadata: ModelMetadata,
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
    pub(super) fn load_and_describe(hef_path: &str) -> Result<Self> {
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
