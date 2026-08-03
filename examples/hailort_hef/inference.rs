use anyhow::{Context, Result, bail};

use super::model::ModelMetadata;
use super::runtime::HailoVStreams;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct InferenceContract {
    pub(super) input_frame_bytes: usize,
    pub(super) output_frame_bytes: usize,
}

pub(super) fn derive_inference_contract(metadata: &ModelMetadata) -> Result<InferenceContract> {
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

pub(super) fn validate_inference_contract(
    metadata: &ModelMetadata,
    vstreams: &HailoVStreams,
) -> Result<()> {
    if vstreams.input_count() != 1 {
        bail!(
            "inference pipeline requires exactly one input virtual stream; HEF created {}",
            vstreams.input_count()
        );
    }

    if vstreams.output_count() != 1 {
        bail!(
            "inference pipeline requires exactly one output virtual stream; HEF created {}",
            vstreams.output_count()
        );
    }

    let contract = derive_inference_contract(metadata)?;

    let actual_input_size = vstreams
        .input_frame_size(0)
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
        .output_frame_size(0)
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

pub(super) fn infer_one(vstreams: &HailoVStreams, input_frame: &[u8]) -> Result<Vec<u8>> {
    let input_frame_size = vstreams
        .input_frame_size(0)
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
        .output_frame_size(0)
        .map_err(anyhow::Error::msg)
        .context("failed to obtain the output frame size")?;

    let mut output_frame = vec![0_u8; output_frame_size];

    println!(
        "Writing one preprocessed RGB image: {} bytes",
        input_frame.len()
    );

    vstreams
        .write_input(0, input_frame)
        .map_err(anyhow::Error::msg)
        .context("failed to write the input frame")?;

    println!("Input frame written successfully");
    println!("Reading one inference result");

    vstreams
        .read_output(0, &mut output_frame)
        .map_err(anyhow::Error::msg)
        .context("failed to read the output frame")?;

    println!(
        "Output frame read successfully: {} bytes",
        output_frame.len()
    );

    vstreams
        .flush_input(0)
        .map_err(anyhow::Error::msg)
        .context("failed to flush the input virtual stream")?;

    println!("Input virtual stream flushed successfully");

    Ok(output_frame)
}
