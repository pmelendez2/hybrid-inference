pub(super) fn load_rgb_image(
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

pub(super) struct PreprocessedImage {
    pub(super) pixels: Vec<u8>,
    pub(super) original_width: u32,
    pub(super) original_height: u32,
    pub(super) resized_width: u32,
    pub(super) resized_height: u32,
    pub(super) pad_x: u32,
    pub(super) pad_y: u32,
}
