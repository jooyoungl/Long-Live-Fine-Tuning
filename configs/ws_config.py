# CONFIG for the environment Dataset
from pathlib import Path

# ========== DIRECTORIES ==========
# Water Scandal: Input / Output
INPUT_DIR = Path("input/water_scandal")
OUTPUT_DIR = Path("output/water_scandal/evaluated")
OUTPUT_JSON = Path("output/water_scandal/json/raw")
OUTPUT_METRICS = Path("output/water_scandal/metrics")
CLASSIFIED_DIR = Path("output/water_scandal/json/classified")


# Water Scandal: CSV and Data Files
DATA_CSV = INPUT_DIR / "water_scandal_data.csv"
GOLD_STANDARD_TOPIC = "environment"

# Label set Zero-Shot
LABEL_SET = ["spreads misinformation", "corrects misinformation", "provides neutral or unrelated commentary"] # LABEL SET 1
# LABEL_SET = ['outrage', 'endorse', 'skepticism', 'fact-check', 'off-topic', 'meta'] # LABEL SET 2

# Model directories
ZS_MODEL_NAME = "facebook/bart-large-mnli"      # Zero-shot classification
