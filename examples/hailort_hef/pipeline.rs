use anyhow::{Context, Result};

use super::annotation::{annotate_image, map_detection_to_original};
use super::inference::{infer_one, validate_inference_contract};
use super::model::{LoadedModel, ModelMetadata};
use super::postprocessing::{Detection, class_name, postprocess_yolov8};
use super::preprocessing::{PreprocessedImage, load_rgb_image};
use super::runtime::{HailoActivatedNetworkGroup, HailoConfiguredDevice, HailoVStreams};

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

pub(super) fn run_inference(
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
