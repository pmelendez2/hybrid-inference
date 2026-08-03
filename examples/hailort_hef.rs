use anyhow::{Context, Result};

#[path = "hailort_hef/postprocessing.rs"]
mod postprocessing;

#[path = "hailort_hef/preprocessing.rs"]
mod preprocessing;

#[path = "hailort_hef/runtime.rs"]
mod runtime;

#[path = "hailort_hef/model.rs"]
mod model;

#[path = "hailort_hef/inference.rs"]
mod inference;

#[path = "hailort_hef/annotation.rs"]
mod annotation;

#[path = "hailort_hef/cli.rs"]
mod cli;

use annotation::{annotate_image, map_detection_to_original};
use cli::{Cli, validate_readable_file};
use inference::{infer_one, validate_inference_contract};
use model::{LoadedModel, ModelMetadata};
use postprocessing::{Detection, class_name, postprocess_yolov8};
use preprocessing::{PreprocessedImage, load_rgb_image};
use runtime::{HailoActivatedNetworkGroup, HailoConfiguredDevice, HailoVStreams};

use clap::Parser;

#[allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod hailort {
    include!(concat!(env!("OUT_DIR"), "/hailort_bindings.rs"));
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

    let configured_device = HailoConfiguredDevice::open_and_configure(&model.hef)?;
    let network_group = configured_device.first_network_group()?;

    let vstreams = HailoVStreams::create_and_describe(network_group)?;
    validate_inference_contract(&model.metadata, &vstreams)?;

    {
        let _activation = HailoActivatedNetworkGroup::activate(network_group)
            .map_err(anyhow::Error::msg)
            .context("failed to activate the network group.")?;

        process_image(
            &vstreams,
            &model.metadata,
            image_path,
            output_path,
            confidence_threshold,
        )?;
    }

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
    use super::annotation::{
        PixelBoundingBox, detection_label_y, format_detection_label, is_valid_bounding_box,
    };
    use super::inference::{InferenceContract, derive_inference_contract};
    use super::*;
    use image::{Rgb, RgbImage};
    use postprocessing::{COCO_CLASS_NAMES, decode_nms_by_class};

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
