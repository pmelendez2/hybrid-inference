use anyhow::{Context, Result, bail};
use chrono::Local;
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::process::Command;
use std::time::{Duration, Instant};

/// Coordinate Ollama language inference with a local Hailo-8 accelerator.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Ollama model to run.
    #[arg(short, long, env = "OLLAMA_MODEL", default_value = "gemma3")]
    model: String,

    /// Base URL of the Ollama server.
    #[arg(
        long,
        env = "OLLAMA_HOST",
        default_value = "http://localhost:11434"
    )]
    ollama_host: String,

    /// Prompt sent to the language model.
    #[arg(
        short,
        long,
        default_value = "Explain how Rust can coordinate a local Ollama language model and a Hailo-8 vision accelerator. Clearly distinguish their roles."
    )]
    prompt: String,

    /// Maximum number of generated tokens.
    #[arg(short = 'n', long, default_value_t = 256)]
    num_predict: u32,

    /// Sampling temperature.
    #[arg(short, long, default_value_t = 0.2)]
    temperature: f32,

    /// Random seed used for repeatable results.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// How long Ollama should keep the model loaded.
    #[arg(long, default_value = "30m")]
    keep_alive: String,
}

#[derive(Debug, Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    keep_alive: String,
    options: GenerateOptions,
}

#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f32,
    seed: u64,
    num_predict: u32,
}

#[derive(Debug, Deserialize)]
struct GenerateChunk {
    #[serde(default)]
    response: String,

    #[serde(default)]
    done: bool,

    #[serde(default)]
    error: Option<String>,

    #[serde(default)]
    total_duration: Option<u64>,

    #[serde(default)]
    load_duration: Option<u64>,

    #[serde(default)]
    prompt_eval_count: Option<u64>,

    #[serde(default)]
    prompt_eval_duration: Option<u64>,

    #[serde(default)]
    eval_count: Option<u64>,

    #[serde(default)]
    eval_duration: Option<u64>,
}

#[derive(Debug, Default)]
struct OllamaMetrics {
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u64>,
    eval_duration: Option<u64>,
}

impl OllamaMetrics {
    fn update(&mut self, chunk: &GenerateChunk) {
        if chunk.total_duration.is_some() {
            self.total_duration = chunk.total_duration;
        }

        if chunk.load_duration.is_some() {
            self.load_duration = chunk.load_duration;
        }

        if chunk.prompt_eval_count.is_some() {
            self.prompt_eval_count = chunk.prompt_eval_count;
        }

        if chunk.prompt_eval_duration.is_some() {
            self.prompt_eval_duration = chunk.prompt_eval_duration;
        }

        if chunk.eval_count.is_some() {
            self.eval_count = chunk.eval_count;
        }

        if chunk.eval_duration.is_some() {
            self.eval_duration = chunk.eval_duration;
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.num_predict == 0 {
        bail!("--num-predict must be greater than zero");
    }

    if !(0.0..=2.0).contains(&cli.temperature) {
        bail!("--temperature must be between 0.0 and 2.0");
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let started_at = Local::now();
    let client_started = Instant::now();

    let model = cli.model.clone();

    let ollama_host = cli.ollama_host.trim_end_matches('/').to_string();

    let prompt = cli.prompt.clone();

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client")?;

    println!("Hybrid AI Test");
    println!("Model:       {model}");
    println!("Ollama host: {ollama_host}");
    println!(
        "Started:     {}",
        started_at.format("%Y-%m-%d %H:%M:%S%.3f %:z")
    );

    check_hailo()?;

    println!("\n=== Direct Hailo-8 Inference ===");
    println!(
        "Device discovery succeeded. Actual inference requires a \
         Hailo-compiled HEF model."
    );

    println!("\n=== Ollama Inference ===");
    println!("Prompt:      {prompt}");
    println!("Max tokens:  {}", cli.num_predict);
    println!("Temperature: {}", cli.temperature);
    println!("Seed:        {}", cli.seed);
    println!("Keep alive:  {}", cli.keep_alive);
    println!("\nResponse:");

    let request = GenerateRequest {
        model: model.clone(),
        prompt: prompt.clone(),
        stream: true,
        keep_alive: cli.keep_alive.clone(),
        options: GenerateOptions {
            temperature: cli.temperature,
            seed: cli.seed,
            num_predict: cli.num_predict,
        },
    };

    let url = format!("{ollama_host}/api/generate");
    let request_started = Instant::now();

    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .with_context(|| format!("failed to send request to {url}"))?;

    let mut response = check_http_status(response).await?;

    let mut pending = Vec::<u8>::new();
    let mut metrics = OllamaMetrics::default();
    let mut first_token_at: Option<Instant> = None;
    let mut received_done = false;

    while let Some(bytes) = response
        .chunk()
        .await
        .context("failed while reading Ollama response stream")?
    {
        pending.extend_from_slice(&bytes);

        while let Some(newline_position) = pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = pending.drain(..=newline_position).collect();

            process_stream_line(
                &line,
                request_started,
                &mut first_token_at,
                &mut metrics,
                &mut received_done,
            )?;
        }
    }

    // Process a final JSON object if the stream did not end with a newline.
    if !pending.is_empty() {
        process_stream_line(
            &pending,
            request_started,
            &mut first_token_at,
            &mut metrics,
            &mut received_done,
        )?;
    }

    println!();

    if !received_done {
        bail!("Ollama response stream ended without a final done=true message");
    }

    let finished_at = Local::now();
    let client_duration = client_started.elapsed();

    print_summary(
        &model,
        started_at,
        finished_at,
        client_duration,
        request_started,
        first_token_at,
        &metrics,
    );

    Ok(())
}

fn check_hailo() -> Result<()> {
    println!("\n=== Hailo-8 Check ===");

    let output = Command::new("hailortcli")
        .arg("scan")
        .output()
        .context("failed to execute hailortcli scan")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.trim().is_empty() {
        println!("{}", stdout.trim());
    }

    if !output.status.success() {
        if !stderr.trim().is_empty() {
            eprintln!("{}", stderr.trim());
        }

        bail!("hailortcli scan failed with status {}", output.status);
    }

    Ok(())
}

async fn check_http_status(response: reqwest::Response) -> Result<reqwest::Response> {
    let status = response.status();

    if status.is_success() {
        return Ok(response);
    }

    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("unable to read error response: {error}"));

    bail!("Ollama returned HTTP {status}: {body}");
}

fn process_stream_line(
    line: &[u8],
    request_started: Instant,
    first_token_at: &mut Option<Instant>,
    metrics: &mut OllamaMetrics,
    received_done: &mut bool,
) -> Result<()> {
    let line = std::str::from_utf8(line)
        .context("Ollama returned non-UTF-8 data")?
        .trim();

    if line.is_empty() {
        return Ok(());
    }

    let chunk: GenerateChunk =
        serde_json::from_str(line).with_context(|| format!("invalid Ollama JSON: {line}"))?;

    if let Some(error) = chunk.error.as_deref() {
        bail!("Ollama generation error: {error}");
    }

    if !chunk.response.is_empty() {
        if first_token_at.is_none() {
            *first_token_at = Some(Instant::now());
        }

        print!("{}", chunk.response);
        io::stdout()
            .flush()
            .context("failed to flush generated output")?;
    }

    metrics.update(&chunk);

    if chunk.done {
        *received_done = true;
    }

    // Keep the parameter here because it makes the timing relationship explicit.
    let _ = request_started;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_summary(
    model: &str,
    started_at: chrono::DateTime<Local>,
    finished_at: chrono::DateTime<Local>,
    client_duration: Duration,
    request_started: Instant,
    first_token_at: Option<Instant>,
    metrics: &OllamaMetrics,
) {
    println!("\n=== Summary ===");
    println!("Model:                 {model}");
    println!(
        "Started:               {}",
        started_at.format("%Y-%m-%d %H:%M:%S%.3f %:z")
    );
    println!(
        "Finished:              {}",
        finished_at.format("%Y-%m-%d %H:%M:%S%.3f %:z")
    );

    match first_token_at {
        Some(first_token) => {
            println!(
                "Time to first token:    {:.3} seconds",
                first_token.duration_since(request_started).as_secs_f64()
            );
        }
        None => println!("Time to first token:    unavailable"),
    }

    println!(
        "Client duration:        {:.3} seconds",
        client_duration.as_secs_f64()
    );

    print_nanoseconds("Ollama total duration:", metrics.total_duration);
    print_nanoseconds("Model load duration:", metrics.load_duration);

    match metrics.prompt_eval_count {
        Some(count) => println!("Prompt tokens:          {count}"),
        None => println!("Prompt tokens:          unavailable"),
    }

    print_nanoseconds("Prompt evaluation:", metrics.prompt_eval_duration);

    match metrics.eval_count {
        Some(count) => println!("Generated tokens:       {count}"),
        None => println!("Generated tokens:       unavailable"),
    }

    match (metrics.eval_count, metrics.eval_duration) {
        (Some(tokens), Some(duration_ns)) if duration_ns > 0 => {
            let seconds = nanoseconds_to_seconds(duration_ns);
            let tokens_per_second = tokens as f64 / seconds;

            println!("Generation speed:       {tokens_per_second:.2} tokens/second");
        }
        _ => println!("Generation speed:       unavailable"),
    }
}

fn print_nanoseconds(label: &str, value: Option<u64>) {
    match value {
        Some(nanoseconds) => {
            println!(
                "{label:<23} {:.3} seconds",
                nanoseconds_to_seconds(nanoseconds)
            );
        }
        None => println!("{label:<23} unavailable"),
    }
}

fn nanoseconds_to_seconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000_000.0
}
