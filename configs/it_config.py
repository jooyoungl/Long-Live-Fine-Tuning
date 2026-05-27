# CONFIG for the Immigration Dataset
from pathlib import Path

# ========== DIRECTORIES ==========
# Immigration Tattoos: Input / Output
INPUT_DIR = Path("input/immigration_tattoos")
OUTPUT_DIR = Path("output/immigration_tattoos/evaluated")
OUTPUT_JSON = Path("output/immigration_tattoos/json/raw")
OUTPUT_METRICS = Path("output/immigration_tattoos/metrics")
CLASSIFIED_DIR = Path("output/immigration_tattoos/json/classified")

# Immigration Tattoos: CSV and Data Files
DATA_CSV = INPUT_DIR / "immigration_tattoos_data.csv"
GOLD_STANDARD_TOPIC = "immigration"

# Label set Zero-Shot
LABEL_SET = ["spreads misinformation", "corrects misinformation", "provides neutral or unrelated commentary"] # LABEL SET 1
# LABEL_SET = ['criminal', 'insistence', 'clarification', 'photoshop', 'political', 'meta'] # LABEL SET 2

# Model directories
ZS_MODEL_NAME = "facebook/bart-large-mnli"      # Zero-shot classification

