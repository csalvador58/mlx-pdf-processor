# src/config.py
"""
Configuration settings for PDF processing.
Contains all configurable parameters and model settings.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
print(f"\n=== Configuration Initialization ===")
print(f"Looking for .env at: {env_path}")
print(f".env file exists: {env_path.exists()}\n")

load_dotenv(env_path, override=True)

# Directories and file settings
INPUT_DIRECTORY = "pdfs"                  # Directory containing PDF files to process
OUTPUT_DIRECTORY = "output"               # Directory for JSON output files
LOG_DIRECTORY = "log"                     # Directory for log files
OUTPUT_FILE_FORMAT = "{filename}.json"    # Format for output filenames

# Processing settings
TIMEOUT = int(os.getenv("TIMEOUT", 10))  # Timeout period between processing files

# MLX model settings
DEFAULT_LLM_MODEL = "mlx-community/Mistral-Small-24B-Instruct-2501-4bit"
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)  # Default model path
if not LLM_MODEL:
    LLM_MODEL = DEFAULT_LLM_MODEL
    logging.warning(f"LLM_MODEL was empty \n Will use a default model: {LLM_MODEL} \n Model will be downloaded from Hugging Face unless already exists in path ~/.cache/huggingface/hub directory")
    
ADAPTER_PATH = None  # Optional path for trained adapter weights
PREFILL_RESPONSE = None                  # Prefill response for chat template

# Generation settings
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))                    # Maximum tokens to generate
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))                 # Sampling temperature
TOP_P = float(os.getenv("TOP_P", 1.0))                           # Sampling top-p
MIN_P = float(os.getenv("MIN_P", 0.0))                           # Sampling min-p
MIN_TOKENS_TO_KEEP = int(os.getenv("MIN_TOKENS_TO_KEEP", 1))     # Minimum tokens to keep for min-p sampling
SEED = None                              # PRNG seed

# Advanced settings
IGNORE_CHAT_TEMPLATE = False             # Use raw prompt without tokenizer's chat template
USE_DEFAULT_CHAT_TEMPLATE = False        # Use default chat template
CHAT_TEMPLATE_CONFIG = None              # Additional config for apply_chat_template
VERBOSE = True                           # Log verbose output
MAX_KV_SIZE = None                       # Maximum key-value cache size
PROMPT_CACHE_FILE = None                 # File with saved KV caches
KV_BITS = None                           # Number of bits for KV cache quantization
KV_GROUP_SIZE = 64                       # Group size for KV cache quantization
QUANTIZED_KV_START = 5000                # When to start quantizing the KV cache
DRAFT_MODEL = None                       # Model for speculative decoding
NUM_DRAFT_TOKENS = 3                     # Number of tokens to draft for speculative decoding
EXTRA_EOS_TOKEN = []                     # Additional end-of-sequence tokens

# Prompt template 
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a system with three phases:
Phase 1:
- Analyze the pdf file and determine key: value relationships. Focus on extracting data as accurately as possible, even if characters appear unusual.
Phase 2:
- Transform the key: value relationships into a flat JSON object.
- Use underscore notation to represent hierarchical relationships (e.g., subscription_amount instead of subscription.amount)
- Ensure all keys are at the root level with no nested objects or arrays
Phase 3:
- **Critical Accuracy Check:** Compare the generated JSON object with the raw PDF text. Specifically:
 - **Encoding Correction:** Look for replacement characters (like "�") that likely represent incorrectly decoded dashes ("-"). Replace these instances.
 - **Dash Replacement:** If you find a replacement character in fields similar to `invoice_number`, `receipt_number`, or addresses, attempt to replace it with a dash ("-") if contextually appropriate. Prioritize this correction.
 - **General Verification:** Ensure all numerical values (dates, amounts) are correctly represented and that no characters have been altered unintentionally.
The final response should be a single-level JSON object with no nested structures.""")
PROMPT_TEMPLATE = """{pdf_content}"""