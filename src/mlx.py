# src/mlx.py
"""
MLX model operations.
Handles model loading and text generation.
"""

import time
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from . import config
from . import utils


def run_mlx_generate(model, tokenizer, pdf_content):
    """Run the MLX generate function with the configured parameters."""
    prompt = config.PROMPT_TEMPLATE.format(pdf_content=pdf_content)
    
    # Prepare messages for chat template if needed
    if config.SYSTEM_PROMPT and not config.IGNORE_CHAT_TEMPLATE and tokenizer.chat_template is not None:
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": prompt})
        
        has_prefill = config.PREFILL_RESPONSE is not None
        if has_prefill:
            messages.append({"role": "assistant", "content": config.PREFILL_RESPONSE})
            
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
        )
    
    # Configure sampler based on temperature, top_p, etc.
    sampler = make_sampler(
        config.TEMPERATURE, 
        config.TOP_P, 
        config.MIN_P, 
        config.MIN_TOKENS_TO_KEEP
    )
    
    # Set random seed if provided
    if config.SEED is not None:
        mx.random.seed(config.SEED)
    
    # Start timing the LLM generation
    llm_generate_start = time.time()
    
    # Capture stdout to extract metrics
    with utils.OutputCapture() as output:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=config.MAX_TOKENS,
            verbose=config.VERBOSE,
            sampler=sampler,
            max_kv_size=config.MAX_KV_SIZE,
            prompt_cache=None,
            kv_bits=config.KV_BITS,
            kv_group_size=config.KV_GROUP_SIZE,
            quantized_kv_start=config.QUANTIZED_KV_START,
            draft_model=config.DRAFT_MODEL,
            num_draft_tokens=config.NUM_DRAFT_TOKENS
        )
    
    # Calculate LLM generation time
    llm_generate_time = time.time() - llm_generate_start
    llm_generate_time_formatted = utils.format_time_hh_mm_ss(llm_generate_time)
    
    # Parse metrics from the captured output
    metrics = utils.parse_generate_output(output.get_output())
    
    # Combine response text with metrics
    result = {
        "llm_response": response,
        "llm_prompt": metrics["llm_prompt"],
        "llm_generation": metrics["llm_generation"], 
        "llm_peak_memory": metrics["llm_peak_memory"],
        "tt_llm_generate": llm_generate_time_formatted
    }
    
    return result