# src/processor.py
"""
Core PDF processing logic.
Manages the workflow of converting and processing PDFs.
"""

import time
from pathlib import Path
from mlx_lm import load

from . import config
from . import utils
from . import mlx


def process_pdf_files():
    """Process all PDF files in the input directory."""
    # Start timing overall execution
    overall_start_time = time.time()
    
    utils.ensure_directory_exists(config.INPUT_DIRECTORY)
    utils.ensure_directory_exists(config.OUTPUT_DIRECTORY)
    utils.ensure_directory_exists(config.LOG_DIRECTORY)
    
    # Get list of PDF files
    pdf_files = list(Path(config.INPUT_DIRECTORY).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {config.INPUT_DIRECTORY}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    # Load model and tokenizer once for all PDFs
    print(f"Loading model from {config.LLM_MODEL}...")
    model, tokenizer = load(
        path_or_hf_repo=config.LLM_MODEL,
        adapter_path=config.ADAPTER_PATH,
        tokenizer_config={"trust_remote_code": True}
    )
    print("Model loaded successfully")
    
    # Process each PDF file
    for i, pdf_path in enumerate(pdf_files):
        
        # print(f"Processing PDF: {pdf_path} ({i+1}/{len(pdf_files)})...")
        
        # Get base filename without extension
        filename = pdf_path.stem
        
        # Read PDF content as markdown
        pdf_content = utils.read_pdf_as_markdown(pdf_path)
        if not pdf_content:
            print(f"Skipping {pdf_path} due to empty content")
            continue
        
        # Run MLX generate function
        response_data = mlx.run_mlx_generate(model, tokenizer, pdf_content)
        
        # Calculate and add overall time
        current_overall_time = time.time() - overall_start_time
        current_overall_time_formatted = utils.format_time_hh_mm_ss(current_overall_time)
        response_data["tt_overall"] = current_overall_time_formatted
        
        # Log timing information
        print(f"✅ Completed {filename}")
        print(f"⏱️ LLM generation time: {response_data['tt_llm_generate']}")
        print(f"⏱️ Elapsed time: {current_overall_time_formatted}")
        
        # Save output
        utils.save_output(filename, response_data)
        
        # Apply timeout between files (except after the last file)
        if i < len(pdf_files) - 1 and config.TIMEOUT > 0:
            print(f"Cooling down for {config.TIMEOUT} seconds...")
            time.sleep(config.TIMEOUT)
    
    # Calculate final overall time
    final_overall_time = time.time() - overall_start_time
    final_overall_time_formatted = utils.format_time_hh_mm_ss(final_overall_time)
    print(f"\nAll PDF files processed successfully!")
    print(f"Total time: {final_overall_time_formatted}")