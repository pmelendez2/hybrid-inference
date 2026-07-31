use std::path::Path;

use ab_glyph::FontRef;
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;

use super::PreprocessedImage;
use super::postprocessing::{Detection, class_name};

const ANNOTATION_FONT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/annotation-font.ttf"
));

const ANNOTATION_COLOR: Rgb<u8> = Rgb([255, 0, 0]);
const ANNOTATION_BOX_THICKNESS: u32 = 3;
const ANNOTATION_LABEL_OFFSET: u32 = 28;
const ANNOTATION_LABEL_FONT_SIZE: f32 = 24.0;

#[derive(Debug)]
pub(super) struct PixelBoundingBox {
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) right: u32,
    pub(super) bottom: u32,
}

pub(super) fn is_valid_bounding_box(bounding_box: &PixelBoundingBox) -> bool {
    bounding_box.left < bounding_box.right && bounding_box.top < bounding_box.bottom
}

pub(super) fn detection_label_y(pixel_box: &PixelBoundingBox) -> u32 {
    pixel_box.top.saturating_sub(ANNOTATION_LABEL_OFFSET)
}

pub(super) fn map_detection_to_original(
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

    for thickness in 0..ANNOTATION_BOX_THICKNESS {
        let left = pixel_box.left.saturating_add(thickness);
        let top = pixel_box.top.saturating_add(thickness);

        let box_width = width.saturating_sub(thickness * 2).max(1);

        let box_height = height.saturating_sub(thickness * 2).max(1);

        let rectangle = Rect::at(left as i32, top as i32).of_size(box_width, box_height);

        draw_hollow_rect_mut(annotated, rectangle, ANNOTATION_COLOR);
    }
}

pub(super) fn format_detection_label(detection: &Detection) -> String {
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

pub(super) fn annotate_image(
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

    save_annotated_image(&annotated, output_path)
}
