use anyhow::{Context, Result};

pub(super) const COCO_CLASS_NAMES: [&str; 80] = [
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

pub(super) fn class_name(class_id: usize) -> &'static str {
    COCO_CLASS_NAMES.get(class_id).copied().unwrap_or("unknown")
}

#[derive(Debug)]
pub(super) struct Detection {
    pub(super) class_id: usize,
    pub(super) confidence: f32,
    pub(super) ymin: f32,
    pub(super) xmin: f32,
    pub(super) ymax: f32,
    pub(super) xmax: f32,
}

pub(super) fn decode_nms_by_class(
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

pub(super) fn postprocess_yolov8(
    output_frame: &[u8],
    confidence_threshold: f32,
    class_count: usize,
    max_detections_per_class: usize,
) -> anyhow::Result<Vec<Detection>> {
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
