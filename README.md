# MLX PDF Processor

Processes PDF files using MLX LM. Converts PDFs to markdown format, LLM processing, and outputs structured JSON results.

## Features

- PDF to markdown conversion with [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html)
- Extracts text from PDFs and generate a flattened JSON output

## Requirements

- Python 3.11 or higher
- MLX-LM
- pymupdf4llm

## Installation

1. Clone the repository
2. Install using your preferred Python package manager:

```bash
# Using uv (recommended)
uv sync

# Run the processor
uv run main.py

# Or using command line arguments
uv run main.py --input-dir custom_pdfs --timeout 60
```

## Configuration

Configure via environment variables or .env file:

- `LLM_MODEL`: Model path (default: mlx-community/Mistral-Small-24B-Instruct-2501-4bit)
- `MAX_TOKENS`: Maximum tokens to generate
- `TEMPERATURE`: Sampling temperature
- `SYSTEM_PROMPT`: Custom system prompt
- Additional settings in config.py

## Usage

```bash
python main.py [options]

Options:
  --input-dir DIR    Input directory (default: pdfs)
  --output-dir DIR   Output directory (default: output)
  --log-dir DIR      Log directory (default: log)
  --model PATH       MLX model path
  --timeout SEC      Timeout between files
```

## Output Structure

Each processed PDF generates:
- JSON file with configuration, metrics, and LLM response
- Text log file with raw model output
- Console progress updates with timing information

## About MLX-LM

This project uses [MLX-LM](https://github.com/ml-explore/mlx-lm), a Python package for text generation and LLM fine-tuning on Apple silicon with MLX. MLX-LM provides:

For more details about MLX-LM and its capabilities, visit the [official repository](https://github.com/ml-explore/mlx-lm).