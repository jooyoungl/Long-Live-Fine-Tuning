# CONFIG for the Health Dataset
from pathlib import Path

# ========== DIRECTORIES ==========
# Trump Bleach: Input / Output
INPUT_DIR = Path("input/trump_bleach")
OUTPUT_DIR = Path("output/trump_bleach/evaluated")
OUTPUT_JSON = Path("output/trump_bleach/json/raw")
OUTPUT_METRICS = Path("output/trump_bleach/metrics")
CLASSIFIED_DIR = Path("output/trump_bleach/json/classified")

# Trump Bleach: CSV and Data Files
DATA_CSV = INPUT_DIR / "trump_bleach_data.csv"
GOLD_STANDARD_TOPIC = "health"

LABEL_SET = ["spreads misinformation", "corrects misinformation", "provides neutral or unrelated commentary"] # LABEL_SET 1
# LABEL_SET = ['disbelief', 'insistence', 'clarification', 'skepticism', 'political', 'meta'] # LABEL_SET 2

# Model directories
ZS_MODEL_NAME = "facebook/bart-large-mnli"      # Zero-shot classification
