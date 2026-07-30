use std::fs::File;

use anyhow::{Context, Result, bail};
use clap::Parser;

const DEFAULT_HEF_PATH: &str = "/models/hefs/yolov8n_hailo8_v2.18.0.hef";
const DEFAULT_IMAGE_PATH: &str = "/models/images/culdesac-latest.jpg";
const DEFAULT_OUTPUT_PATH: &str = "output/annotated.jpg";
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.50;

fn parse_confidence_threshold(value: &str) -> Result<f32, String> {
    let threshold = value
        .parse::<f32>()
        .map_err(|error| format!("invalid confidence threshold '{value}': {error}"))?;

    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!(
            "confidence threshold must be between 0.0 and 1.0; \
             received {value}"
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
pub(super) struct Cli {
    /// Path to the Hailo executable format model
    #[arg(long, value_name = "FILE", default_value = DEFAULT_HEF_PATH)]
    pub(super) hef: String,

    /// Path to the input image
    #[arg(short, long, value_name = "FILE", default_value = DEFAULT_IMAGE_PATH)]
    pub(super) input: String,

    /// Path for the annotated output image
    #[arg(short, long, value_name = "FILE", default_value = DEFAULT_OUTPUT_PATH)]
    pub(super) output: String,

    /// Minimum detection confidence from 0.0 through 1.0
    #[arg(
        short,
        long,
        value_name = "NUMBER",
        default_value_t = DEFAULT_CONFIDENCE_THRESHOLD,
        value_parser = parse_confidence_threshold
    )]
    pub(super) confidence: f32,
}

pub(super) fn validate_readable_file(path: &str, description: &str) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("{description} '{path}' cannot be accessed"))?;

    if !metadata.is_file() {
        bail!("{description} '{path}' is not a regular file");
    }

    File::open(path)
        .with_context(|| format!("{description} '{path}' cannot be opened for reading"))?;

    Ok(())
}
