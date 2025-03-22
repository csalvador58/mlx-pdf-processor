#!/usr/bin/env python3
"""
Entry point for MLX PDF processor.
Handles CLI arguments and initiates processing.
"""

import argparse
from src import config, processor


def main():
    """Main function to run the script."""
    # Parse command line arguments to override config
    parser = argparse.ArgumentParser(description="Process PDF files with MLX")
    parser.add_argument("--input-dir", help=f"Input directory (default: {config.INPUT_DIRECTORY})")
    parser.add_argument("--output-dir", help=f"Output directory (default: {config.OUTPUT_DIRECTORY})")
    parser.add_argument("--log-dir", help=f"Log directory (default: {config.LOG_DIRECTORY})")
    parser.add_argument("--model", help=f"MLX model (default: {config.LLM_MODEL})")
    parser.add_argument("--timeout", type=int, help=f"Timeout seconds between files (default: {config.TIMEOUT})")
    args = parser.parse_args()
    
    # Override configuration if provided
    if args.input_dir:
        config.INPUT_DIRECTORY = args.input_dir
    if args.output_dir:
        config.OUTPUT_DIRECTORY = args.output_dir
    if args.log_dir:
        config.LOG_DIRECTORY = args.log_dir
    if args.model:
        config.LLM_MODEL = args.model
    if args.timeout is not None:
        config.TIMEOUT = args.timeout
    
    # Process PDF files
    processor.process_pdf_files()


if __name__ == "__main__":
    main()