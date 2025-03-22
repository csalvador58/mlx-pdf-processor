# src/utils.py
"""
Utility functions for PDF processing.
Contains helpers for file operations, logging, and output formatting.
"""

import io
import json
import re
import sys
from pathlib import Path
import pymupdf4llm

from . import config


def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist."""
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True)
        print(f"Created directory: {directory_path}")
    return path


def read_pdf_as_markdown(pdf_path):
    """Read PDF and convert to markdown using pymupdf4llm."""
    try:
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        return markdown_text
    except Exception as e:
        print(f"Error converting PDF {pdf_path} to markdown: {e}")
        return ""


class OutputCapture:
    """Capture stdout to extract metrics from verbose output."""
    def __init__(self):
        self.original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        
    def __enter__(self):
        sys.stdout = self.captured_output
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        
    def get_output(self):
        return self.captured_output.getvalue()


def parse_generate_output(output_text):
    """Parse the stdout from generate to extract metrics."""
    response_data = {
        "llm_prompt": "",
        "llm_generation": "",
        "llm_peak_memory": ""
    }
    
    # Extract metrics using regex
    prompt_match = re.search(r'Prompt: (.*)', output_text)
    if prompt_match:
        response_data["llm_prompt"] = prompt_match.group(1).strip()
    
    generation_match = re.search(r'Generation: (.*)', output_text)
    if generation_match:
        response_data["llm_generation"] = generation_match.group(1).strip()
    
    peak_memory_match = re.search(r'Peak memory: (.*)', output_text)
    if peak_memory_match:
        response_data["llm_peak_memory"] = peak_memory_match.group(1).strip()
    
    return response_data


def format_time_hh_mm_ss(seconds):
    """Format seconds into HH:MM:SS format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def save_output(filename, response_data):
    """Save the response data to JSON file."""
    # Create output data including all configuration
    output_data = {
        # Include all configuration parameters (except cooldown)
        "config": {
            "input_directory": config.INPUT_DIRECTORY,
            "output_directory": config.OUTPUT_DIRECTORY,
            "log_directory": config.LOG_DIRECTORY,
            "model": config.LLM_MODEL,
            "adapter_path": config.ADAPTER_PATH,
            "system_prompt": config.SYSTEM_PROMPT,
            "prefill_response": config.PREFILL_RESPONSE,
            "max_tokens": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "min_p": config.MIN_P,
            "min_tokens_to_keep": config.MIN_TOKENS_TO_KEEP,
            "seed": config.SEED,
            "ignore_chat_template": config.IGNORE_CHAT_TEMPLATE,
            "use_default_chat_template": config.USE_DEFAULT_CHAT_TEMPLATE,
            "chat_template_config": config.CHAT_TEMPLATE_CONFIG,
            "verbose": config.VERBOSE,
            "max_kv_size": config.MAX_KV_SIZE,
            "prompt_cache_file": config.PROMPT_CACHE_FILE,
            "kv_bits": config.KV_BITS,
            "kv_group_size": config.KV_GROUP_SIZE,
            "quantized_kv_start": config.QUANTIZED_KV_START,
            "draft_model": config.DRAFT_MODEL,
            "num_draft_tokens": config.NUM_DRAFT_TOKENS,
            "extra_eos_token": config.EXTRA_EOS_TOKEN,
            "prompt_template": config.PROMPT_TEMPLATE
        },
        # Include response data
        "llm_response": response_data["llm_response"],
        "llm_prompt": response_data["llm_prompt"],
        "llm_generation": response_data["llm_generation"],
        "llm_peak_memory": response_data["llm_peak_memory"],
        "tt_llm_generate": response_data["tt_llm_generate"],
        "tt_overall": response_data["tt_overall"]
    }
    
    # Save to output file
    output_path = Path(config.OUTPUT_DIRECTORY) / config.OUTPUT_FILE_FORMAT.format(filename=filename)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save raw response to log file
    log_path = Path(config.LOG_DIRECTORY) / f"{filename}.txt"
    with open(log_path, 'w') as f:
        f.write(response_data["llm_response"])
    
    print(f"\nSaved output file: {output_path} \nlog file: {log_path}")