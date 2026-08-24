use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io;
use std::num::ParseFloatError;

use clap::Parser;

const DEFAULT_HEF_PATH: &str = "/models/hefs/yolov8n_hailo8_v2.18.0.hef";
const DEFAULT_IMAGE_PATH: &str = "/models/images/culdesac-latest.jpg";
const DEFAULT_OUTPUT_PATH: &str = "output/annotated.jpg";
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.50;

#[derive(Debug)]
pub(super) enum CliError {
    InvalidConfidenceThreshold {
        value: String,
        source: ParseFloatError,
    },
    ConfidenceThresholdOutOfRange {
        value: String,
    },
    FileCannotBeAccessed {
        description: String,
        path: String,
        source: io::Error,
    },
    NotARegularFile {
        description: String,
        path: String,
    },
    FileCannotBeOpenedForReading {
        description: String,
        path: String,
        source: io::Error,
    },
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfidenceThreshold { value, source } => {
                write!(
                    formatter,
                    "invalid confidence threshold '{value}': {source}"
                )
            }
            Self::ConfidenceThresholdOutOfRange { value } => write!(
                formatter,
                "confidence threshold must be between 0.0 and 1.0; received {value}"
            ),
            Self::FileCannotBeAccessed {
                description,
                path,
                source,
            } => write!(
                formatter,
                "{description} '{path}' cannot be accessed: {source}"
            ),
            Self::NotARegularFile { description, path } => {
                write!(formatter, "{description} '{path}' is not a regular file")
            }
            Self::FileCannotBeOpenedForReading {
                description,
                path,
                source,
            } => write!(
                formatter,
                "{description} '{path}' cannot be opened for reading: {source}"
            ),
        }
    }
}

impl Error for CliError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfidenceThreshold { source, .. } => Some(source),
            Self::FileCannotBeAccessed { source, .. }
            | Self::FileCannotBeOpenedForReading { source, .. } => Some(source),
            Self::ConfidenceThresholdOutOfRange { .. } | Self::NotARegularFile { .. } => None,
        }
    }
}

fn parse_confidence_threshold(value: &str) -> Result<f32, CliError> {
    let threshold =
        value
            .parse::<f32>()
            .map_err(|source| CliError::InvalidConfidenceThreshold {
                value: value.to_owned(),
                source,
            })?;

    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(CliError::ConfidenceThresholdOutOfRange {
            value: value.to_owned(),
        });
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

pub(super) fn validate_readable_file(path: &str, description: &str) -> Result<(), CliError> {
    let metadata = std::fs::metadata(path).map_err(|source| CliError::FileCannotBeAccessed {
        description: description.to_owned(),
        path: path.to_owned(),
        source,
    })?;

    if !metadata.is_file() {
        return Err(CliError::NotARegularFile {
            description: description.to_owned(),
            path: path.to_owned(),
        });
    }

    File::open(path).map_err(|source| CliError::FileCannotBeOpenedForReading {
        description: description.to_owned(),
        path: path.to_owned(),
        source,
    })?;

    Ok(())
}
