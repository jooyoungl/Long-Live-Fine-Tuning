import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import pandas as pd
import psutil
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import pipeline
import ollama

# Optional GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ==============================
# SETTINGS
# ==============================

GOLD_CSV = Path("input/gold_standard_data_annotated.csv")
OUTPUT_DIR = Path("comparison_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If you want to filter by topic, set this to a string like "Environment"
# Otherwise leave as None
FILTER_TOPIC = None

TEXT_COLUMN = "comment"
GOLD_LABEL_COLUMN = "final_label"

# BART model
BART_MODEL_NAME = "facebook/bart-large-mnli"

# Ollama models to compare
OLLAMA_MODELS = [
    #"llama3.2:3b",
    "llama3:8b",
    #"llama3:70b",
]

# Candidate labels shown to both BART and LLMs
LABEL_SET = [
    "spreads misinformation",
    "corrects misinformation",
    "provides neutral or unrelated commentary",
]

# Map model outputs to final evaluation labels
LABEL_MAPPING = {
    "spreads misinformation": "belief",
    "corrects misinformation": "fact-check",
    "provides neutral or unrelated commentary": "other",
    "other": "other",
}

FINAL_LABELS = ["belief", "fact-check", "other"]

# Number of warmup examples before measured run
WARMUP_EXAMPLES = 3

# Sampling interval for system metrics during each inference call
MONITOR_INTERVAL_SEC = 0.05

# Set True if you want to save per-example raw monitor summaries
SAVE_PER_EXAMPLE = True


# ==============================
# GPU MONITOR
# ==============================

class GPUTracker:
    def __init__(self):
        self.enabled = False
        self.handle = None

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.enabled = True
            except Exception:
                self.enabled = False

    def snapshot(self) -> Dict[str, Optional[float]]:
        if not self.enabled:
            return {
                "gpu_util_percent": None,
                "gpu_mem_used_gb": None,
                "gpu_power_w": None,
            }

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            except Exception:
                power = None

            return {
                "gpu_util_percent": float(util.gpu),
                "gpu_mem_used_gb": float(mem.used) / (1024 ** 3),
                "gpu_power_w": power,
            }
        except Exception:
            return {
                "gpu_util_percent": None,
                "gpu_mem_used_gb": None,
                "gpu_power_w": None,
            }


# ==============================
# RESOURCE MONITOR
# ==============================

class ResourceMonitor:
    def __init__(self, interval_sec: float = 0.05):
        self.interval_sec = interval_sec
        self.gpu = GPUTracker()
        self._stop_event = threading.Event()
        self._thread = None
        self.samples = []

    def _collect_loop(self):
        process = psutil.Process()
        while not self._stop_event.is_set():
            try:
                ram_gb = process.memory_info().rss / (1024 ** 3)
            except Exception:
                ram_gb = None

            try:
                cpu_percent = process.cpu_percent(interval=None)
            except Exception:
                cpu_percent = None

            gpu_stats = self.gpu.snapshot()

            self.samples.append({
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "ram_used_gb": ram_gb,
                "gpu_util_percent": gpu_stats["gpu_util_percent"],
                "gpu_mem_used_gb": gpu_stats["gpu_mem_used_gb"],
                "gpu_power_w": gpu_stats["gpu_power_w"],
            })
            time.sleep(self.interval_sec)

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

        if not self.samples:
            return {
                "cpu_avg_percent": None,
                "cpu_peak_percent": None,
                "ram_avg_gb": None,
                "ram_peak_gb": None,
                "gpu_util_avg_percent": None,
                "gpu_util_peak_percent": None,
                "gpu_mem_avg_gb": None,
                "gpu_mem_peak_gb": None,
                "gpu_power_avg_w": None,
                "gpu_power_peak_w": None,
            }

        df = pd.DataFrame(self.samples)

        def safe_mean(col):
            s = df[col].dropna()
            return float(s.mean()) if not s.empty else None

        def safe_max(col):
            s = df[col].dropna()
            return float(s.max()) if not s.empty else None

        return {
            "cpu_avg_percent": safe_mean("cpu_percent"),
            "cpu_peak_percent": safe_max("cpu_percent"),
            "ram_avg_gb": safe_mean("ram_used_gb"),
            "ram_peak_gb": safe_max("ram_used_gb"),
            "gpu_util_avg_percent": safe_mean("gpu_util_percent"),
            "gpu_util_peak_percent": safe_max("gpu_util_percent"),
            "gpu_mem_avg_gb": safe_mean("gpu_mem_used_gb"),
            "gpu_mem_peak_gb": safe_max("gpu_mem_used_gb"),
            "gpu_power_avg_w": safe_mean("gpu_power_w"),
            "gpu_power_peak_w": safe_max("gpu_power_w"),
        }


# ==============================
# HELPERS
# ==============================

def normalize_prediction(raw_output: str, labels: List[str]) -> str:
    if raw_output is None:
        return "other"

    output = raw_output.strip().lower()

    # exact match
    for l in labels:
        if output == l.lower():
            return l

    # contained match
    for l in labels:
        if l.lower() in output:
            return l

    return "other"


def map_label(label: str, mapping: Dict[str, str]) -> str:
    return mapping.get(label, "other")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(GOLD_CSV)

    required = {TEXT_COLUMN, GOLD_LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if FILTER_TOPIC is not None:
        if "topic" not in df.columns:
            raise ValueError("FILTER_TOPIC is set but CSV has no 'topic' column.")
        df = df[df["topic"] == FILTER_TOPIC].copy()

    df = df.dropna(subset=[TEXT_COLUMN, GOLD_LABEL_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df[GOLD_LABEL_COLUMN] = df[GOLD_LABEL_COLUMN].astype(str)
    df = df.reset_index(drop=True)
    return df


# ==============================
# BART RUNNER
# ==============================

@dataclass
class BartRunner:
    model_name: str
    classifier: Any = None

    def __post_init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device,
        )

    def predict_with_metrics(self, text: str, labels: List[str]) -> Dict[str, Any]:
        monitor = ResourceMonitor(interval_sec=MONITOR_INTERVAL_SEC)
        start = time.perf_counter()
        monitor.start()

        result = self.classifier(
            text,
            candidate_labels=labels,
            multi_label=False,
        )

        wall_time_sec = time.perf_counter() - start
        resource_stats = monitor.stop()

        pred = result["labels"][0]
        score = float(result["scores"][0])

        return {
            "raw_prediction": pred,
            "raw_score": score,
            "wall_time_sec": wall_time_sec,
            **resource_stats,
        }


# ==============================
# OLLAMA RUNNER
# ==============================

@dataclass
class OllamaRunner:
    model: str

    def predict_with_metrics(self, text: str, labels: List[str]) -> Dict[str, Any]:
        label_block = "\n".join([f"- {l}" for l in labels])

        prompt = (
            "You are a careful annotator.\n"
            "Choose exactly ONE label for the comment.\n\n"
            f"Candidate labels:\n{label_block}\n\n"
            f"Comment:\n{text}\n\n"
            "Return ONLY the exact label text."
        )

        monitor = ResourceMonitor(interval_sec=MONITOR_INTERVAL_SEC)
        start = time.perf_counter()
        monitor.start()

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0},
        )

        wall_time_sec = time.perf_counter() - start
        resource_stats = monitor.stop()

        raw_text = response.get("response", "").strip()

        # Ollama timing fields are typically in nanoseconds
        def ns_to_sec(x):
            return (x / 1e9) if isinstance(x, (int, float)) else None

        return {
            "raw_prediction": raw_text,
            "raw_score": None,
            "wall_time_sec": wall_time_sec,
            "ollama_total_duration_sec": ns_to_sec(response.get("total_duration")),
            "ollama_load_duration_sec": ns_to_sec(response.get("load_duration")),
            "ollama_prompt_eval_duration_sec": ns_to_sec(response.get("prompt_eval_duration")),
            "ollama_eval_duration_sec": ns_to_sec(response.get("eval_duration")),
            "ollama_prompt_eval_count": response.get("prompt_eval_count"),
            "ollama_eval_count": response.get("eval_count"),
            **resource_stats,
        }


# ==============================
# EVALUATION
# ==============================

def evaluate_predictions(df_pred: pd.DataFrame) -> Dict[str, Any]:
    y_true = df_pred["gold_label"].tolist()
    y_pred = df_pred["predicted_label"].tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=FINAL_LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=FINAL_LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=FINAL_LABELS, average="weighted", zero_division=0),
        "belief_f1": report["belief"]["f1-score"],
        "fact_check_f1": report["fact-check"]["f1-score"],
        "other_f1": report["other"]["f1-score"],
        "belief_precision": report["belief"]["precision"],
        "belief_recall": report["belief"]["recall"],
        "fact_check_precision": report["fact-check"]["precision"],
        "fact_check_recall": report["fact-check"]["recall"],
        "other_precision": report["other"]["precision"],
        "other_recall": report["other"]["recall"],
        "n_examples": len(df_pred),
    }

    # Aggregate resource columns if present
    resource_cols = [
        "wall_time_sec",
        "cpu_avg_percent",
        "cpu_peak_percent",
        "ram_avg_gb",
        "ram_peak_gb",
        "gpu_util_avg_percent",
        "gpu_util_peak_percent",
        "gpu_mem_avg_gb",
        "gpu_mem_peak_gb",
        "gpu_power_avg_w",
        "gpu_power_peak_w",
        "ollama_total_duration_sec",
        "ollama_load_duration_sec",
        "ollama_prompt_eval_duration_sec",
        "ollama_eval_duration_sec",
        "ollama_prompt_eval_count",
        "ollama_eval_count",
    ]

    for col in resource_cols:
        if col in df_pred.columns:
            s = df_pred[col].dropna()
            metrics[f"{col}_mean"] = float(s.mean()) if not s.empty else None
            metrics[f"{col}_std"] = float(s.std()) if len(s) > 1 else 0.0 if len(s) == 1 else None
            metrics[f"{col}_max"] = float(s.max()) if not s.empty else None

    return metrics


# ==============================
# CORE RUN
# ==============================

def run_model(
    model_family: str,
    model_name: str,
    df: pd.DataFrame,
    labels: List[str],
    label_mapping: Dict[str, str],
    warmup_examples: int = 0,
) -> pd.DataFrame:
    if model_family == "bart":
        runner = BartRunner(model_name)
    elif model_family == "ollama":
        runner = OllamaRunner(model_name)
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    # Warmup
    for i in range(min(warmup_examples, len(df))):
        _ = runner.predict_with_metrics(df.iloc[i][TEXT_COLUMN], labels)

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        text = row[TEXT_COLUMN]
        gold = row[GOLD_LABEL_COLUMN]

        result = runner.predict_with_metrics(text, labels)
        raw_pred = normalize_prediction(result["raw_prediction"], labels)
        mapped_pred = map_label(raw_pred, label_mapping)

        out_row = {
            "model_family": model_family,
            "model_name": model_name,
            "sample_index": i,
            "text": text,
            "gold_label": gold,
            "raw_prediction": result["raw_prediction"],
            "normalized_prediction": raw_pred,
            "predicted_label": mapped_pred,
            "correct": int(mapped_pred == gold),
            **result,
        }
        rows.append(out_row)

        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"{model_name}: {i + 1}/{total}")

    return pd.DataFrame(rows)


# ==============================
# MAIN
# ==============================

def main():
    df = load_data()
    print(f"Loaded {len(df)} examples from {GOLD_CSV}")

    summary_rows = []

    # --------------------------
    # Run BART
    # --------------------------
    print(f"\nRunning {BART_MODEL_NAME}")
    bart_preds = run_model(
        model_family="bart",
        model_name=BART_MODEL_NAME,
        df=df,
        labels=LABEL_SET,
        label_mapping=LABEL_MAPPING,
        warmup_examples=WARMUP_EXAMPLES,
    )

    bart_pred_path = OUTPUT_DIR / "predictions_bart_UL.csv"
    bart_preds.to_csv(bart_pred_path, index=False)

    bart_metrics = evaluate_predictions(bart_preds)
    bart_metrics["model_family"] = "bart"
    bart_metrics["model_name"] = BART_MODEL_NAME
    bart_metrics["params_label"] = "0.4B"
    summary_rows.append(bart_metrics)

    # --------------------------
    # Run Ollama models
    # --------------------------
    for model_name in OLLAMA_MODELS:
        print(f"\nRunning {model_name}")
        preds = run_model(
            model_family="ollama",
            model_name=model_name,
            df=df,
            labels=LABEL_SET,
            label_mapping=LABEL_MAPPING,
            warmup_examples=WARMUP_EXAMPLES,
        )

        safe_model_name = model_name.replace(":", "_").replace("/", "_")
        pred_path = OUTPUT_DIR / f"predictions_{safe_model_name}_UL.csv"
        preds.to_csv(pred_path, index=False)

        metrics = evaluate_predictions(preds)
        metrics["model_family"] = "ollama"
        metrics["model_name"] = model_name

        if "70b" in model_name.lower():
            metrics["params_label"] = "70B"
        elif "8b" in model_name.lower():
            metrics["params_label"] = "8B"
        elif "3b" in model_name.lower():
            metrics["params_label"] = "3B"
        else:
            metrics["params_label"] = None

        summary_rows.append(metrics)

    # --------------------------
    # Save summary table
    # --------------------------
    summary_df = pd.DataFrame(summary_rows)

    preferred_cols = [
        "model_name",
        "params_label",
        "model_family",
        "n_examples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "belief_f1",
        "fact_check_f1",
        "other_f1",
        "wall_time_sec_mean",
        "wall_time_sec_std",
        "wall_time_sec_max",
        "cpu_avg_percent_mean",
        "cpu_peak_percent_mean",
        "ram_avg_gb_mean",
        "ram_peak_gb_mean",
        "gpu_util_avg_percent_mean",
        "gpu_util_peak_percent_mean",
        "gpu_mem_avg_gb_mean",
        "gpu_mem_peak_gb_mean",
        "gpu_power_avg_w_mean",
        "gpu_power_peak_w_mean",
        "ollama_total_duration_sec_mean",
        "ollama_load_duration_sec_mean",
        "ollama_prompt_eval_duration_sec_mean",
        "ollama_eval_duration_sec_mean",
        "ollama_prompt_eval_count_mean",
        "ollama_eval_count_mean",
    ]

    ordered_cols = [c for c in preferred_cols if c in summary_df.columns] + \
                   [c for c in summary_df.columns if c not in preferred_cols]

    summary_df = summary_df[ordered_cols]

    summary_path = OUTPUT_DIR / "model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Per-model predictions saved under: {OUTPUT_DIR}")
    print(f"Summary table saved to: {summary_path}")


if __name__ == "__main__":
    main()
