use std::ffi::CStr;
use std::fmt;

use super::hailort;
use super::runtime::HailoHef;

#[derive(Debug)]
pub(super) enum ModelError {
    FailedToLoadHef {
        path: String,
        source: String,
    },
    FailedToObtainNetworkGroupInformation {
        source: String,
    },
    NoNetworkGroups,
    FailedToObtainVirtualStreamInformation {
        network_group: String,
        source: String,
    },
    NoNonNmsInputVirtualStream,
    NoNmsByClassOutputVirtualStream,
    InvalidInputShape {
        height: u32,
        width: u32,
        channels: u32,
    },
    UnsupportedInputChannels {
        channels: u32,
    },
    ZeroNmsClasses,
    ZeroMaxDetectionsPerClass,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FailedToLoadHef { path, source } => {
                write!(f, "failed to load HEF '{path}': {source}")
            }
            Self::FailedToObtainNetworkGroupInformation { source } => {
                write!(
                    f,
                    "failed to obtain HEF network-group information: {source}"
                )
            }
            Self::NoNetworkGroups => write!(f, "HEF contains no network groups"),
            Self::FailedToObtainVirtualStreamInformation {
                network_group,
                source,
            } => write!(
                f,
                "failed to obtain virtual-stream information for network group '{network_group}': {source}"
            ),
            Self::NoNonNmsInputVirtualStream => {
                write!(f, "HEF contains no non-NMS input virtual stream")
            }
            Self::NoNmsByClassOutputVirtualStream => {
                write!(
                    f,
                    "HEF contains no HAILO_NMS_BY_CLASS output virtual stream"
                )
            }
            Self::InvalidInputShape {
                height,
                width,
                channels,
            } => write!(
                f,
                "HEF reports an invalid input shape: {height}x{width}x{channels}"
            ),
            Self::UnsupportedInputChannels { channels } => write!(
                f,
                "RGB preprocessing requires 3 input channels, but the HEF reports {channels}"
            ),
            Self::ZeroNmsClasses => write!(f, "HEF reports zero NMS classes"),
            Self::ZeroMaxDetectionsPerClass => {
                write!(f, "HEF reports zero maximum detections per class")
            }
        }
    }
}

impl std::error::Error for ModelError {}
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

fn derive_model_metadata(
    vstreams: &[hailort::hailo_vstream_info_t],
) -> Result<ModelMetadata, ModelError> {
    let input = vstreams
        .iter()
        .find(|vstream| {
            vstream.direction == hailort::hailo_stream_direction_t_HAILO_H2D_STREAM
                && !is_nms_order(vstream.format.order)
        })
        .ok_or(ModelError::NoNonNmsInputVirtualStream)?;

    let output = vstreams
        .iter()
        .find(|vstream| {
            vstream.direction == hailort::hailo_stream_direction_t_HAILO_D2H_STREAM
                && vstream.format.order
                    == hailort::hailo_format_order_t_HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS
        })
        .ok_or(ModelError::NoNmsByClassOutputVirtualStream)?;

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
        return Err(ModelError::InvalidInputShape {
            height: metadata.input_height,
            width: metadata.input_width,
            channels: metadata.input_channels,
        });
    }

    if metadata.input_channels != 3 {
        return Err(ModelError::UnsupportedInputChannels {
            channels: metadata.input_channels,
        });
    }

    if metadata.class_count == 0 {
        return Err(ModelError::ZeroNmsClasses);
    }

    if metadata.max_detections_per_class == 0 {
        return Err(ModelError::ZeroMaxDetectionsPerClass);
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
    pub(super) fn load_and_describe(hef_path: &str) -> Result<Self, ModelError> {
        println!("Loading HEF: {hef_path}");

        let hef = HailoHef::load(hef_path).map_err(|source| ModelError::FailedToLoadHef {
            path: hef_path.to_owned(),
            source,
        })?;

        println!("HEF loaded successfully");

        let network_groups = hef
            .network_groups()
            .map_err(|source| ModelError::FailedToObtainNetworkGroupInformation { source })?;

        let first_network_group = network_groups.first().ok_or(ModelError::NoNetworkGroups)?;

        let first_group_name = network_group_name(first_network_group);

        let vstreams = hef.vstream_infos(&first_group_name).map_err(|source| {
            ModelError::FailedToObtainVirtualStreamInformation {
                network_group: first_group_name.clone(),
                source,
            }
        })?;

        describe_model(&network_groups, &first_group_name, &vstreams);

        let metadata = derive_model_metadata(&vstreams)?;

        Ok(Self { hef, metadata })
    }
}
