#!/usr/bin/env python3

import json
import re
import time
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
from sklearn.metrics import classification_report
import ollama


# ==============================
# SETTINGS
# ==============================

MODEL_NAME = "llama3:8b"   # change if needed
GOLD_CSV = Path("gold_standard_data_annotated.csv")
MAPPING_PATH = Path("configs/label_mapping.json")
OUTPUT_CSV = MODEL_NAME+"_ts_results.csv"


# ==============================
# LABEL SETS
# ==============================

UL = [
    "spreads misinformation",
    "corrects misinformation",
    "provides neutral or unrelated commentary",
]

LABEL_SETS = {
    "ws": {
        "UL": UL,
        "E1": ["outrage", "endorse", "fact-check", "skepticism", "meta", "off-topic"],
    },
    "tb": {
        "UL": UL,
        "H1": ["disbelief", "insistence", "clarification", "skepticism", "meta", "off-topic"],
    },
    "it": {
        "UL": UL,
        "I1": ["criminal", "insistence", "clarification", "photoshop", "meta", "off-topic"],
    },
}

TOPIC_NAME = {
    "ws": "Environment",
    "tb": "Health",
    "it": "Immigration",
}


# ==============================
# OLLAMA RUNNER
# ==============================

@dataclass
class LlamaRunner:
    model: str

    def predict(self, text: str, labels: List[str]) -> str:
        label_block = "\n".join([f"- {l}" for l in labels])

        prompt = (
            "Choose exactly ONE label for the comment.\n\n"
            f"Candidate labels:\n{label_block}\n\n"
            f"Comment:\n{text}\n\n"
            "Output ONLY the exact label text."
        )

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0},
        )

        output = response["response"].strip().lower()

        # exact match
        for l in labels:
            if output == l.lower():
                return l

        # contained match
        for l in labels:
            if l.lower() in output:
                return l

        return "other"


# ==============================
# CORE EXPERIMENT
# ==============================

def run_condition(topic_key, label_set_name, runner, mapping):
    config = importlib.import_module(f"configs.{topic_key}_config")

    df = pd.read_csv(GOLD_CSV)

    # filter gold set by topic
    df = df[df["topic"] == config.GOLD_STANDARD_TOPIC].copy()
    df = df[df["topic"] == config.GOLD_STANDARD_TOPIC].copy().reset_index(drop=True)

    labels = LABEL_SETS[topic_key][label_set_name]

    preds = []
    for i, row in df.iterrows():
        text = str(row["comment"])
        pred = runner.predict(text, labels)
        mapped = mapping.get(pred, pred)
        preds.append(mapped)

        if (i + 1) % 50 == 0:
            print(f"{topic_key}-{label_set_name}: {i+1}/{len(df)}")
 
    y_true = df["final_label"].astype(str).tolist()

    report = classification_report(
        y_true,
        preds,
        labels=["belief", "fact-check", "other"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "Topic": TOPIC_NAME[topic_key],
        "Label Set": label_set_name,
        "Belief F1": report["belief"]["f1-score"],
        "Fact-Check F1": report["fact-check"]["f1-score"],
        "Other F1": report["other"]["f1-score"],
    }


# ==============================
# MAIN
# ==============================

def main():
    mapping = json.loads(MAPPING_PATH.read_text())
    runner = LlamaRunner(MODEL_NAME)

    plan = [
        ("ws", "UL"),
        ("ws", "E1"),
        ("tb", "UL"),
        ("tb", "H1"),
        ("it", "UL"),
        ("it", "I1"),
    ]

    rows = []

    for topic, label_set in plan:
        print(f"\nRunning {topic}-{label_set}")
        result = run_condition(topic, label_set, runner, mapping)
        rows.append(result)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
