"""Central configuration for the MBAX 6418 Assignment 1 pipeline."""
import os

# ---- Paths (relative to project root) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SRC_DIR = os.path.join(ROOT, "src")
OUT_DIR = os.path.join(ROOT, "output")

RAW_JSONL = os.path.join(DATA_DIR, "Gift_Cards.jsonl")
NRC_FILE = os.path.join(DATA_DIR, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

# ---- LLM endpoint (OpenAI-compatible) ----
LLM_BASE_URL = "http://dobolyi.com:9001/v1"
LLM_API_KEY = "6418"
LLM_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

# ---- Reproducibility ----
SEED = 6418           # fixed seed for balanced sampling
BATCH1_N = 100        # step 2: first-batch size
BALANCED_PER_CLASS = 50  # step 6: ~50 per class
MAX_TOKENS = 500
RETRIES = 3
TIMEOUT = 120

# ---- Three-class mapping (rating -> class) ----
def rating_to_three(rating: float) -> str:
    """Step 6 mapping: 4-5 POSITIVE, 3 NEUTRAL, 1-2 NEGATIVE."""
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"

def rating_to_binary(rating: float) -> str:
    """Step 2 mapping: >=4 POSITIVE, else NEGATIVE."""
    return "POSITIVE" if rating >= 4 else "NEGATIVE"
