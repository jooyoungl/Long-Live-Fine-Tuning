#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UL cross-validation fine-tuning for DistilBERT and RoBERTa.

Uses:
- gold_standard_data.csv
- comment
- topic
- assigned_label

Runs:
1) pooled UL CV across all topics
2) per-topic UL CV

Outputs:
- all saved into one flat output folder

Install:
    python -m pip install transformers scikit-learn pandas numpy torch

Example:
    python run_finetune_ul_cv.py \
        --data gold_standard_data.csv \
        --output-dir finetune_ul_cv_outputs \
        --run-pooled \
        --run-per-topic
"""
import time
import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}

DEFAULT_TOPICS = ["environment", "health", "immigration"]


# ============================================================
# UTILS
# ============================================================

def normalize_label(x):
    x = str(x).strip().lower()
    x = x.replace("fact_check", "fact-check")
    x = x.replace("fact check", "fact-check")
    return x

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def safe_name(s):
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("-", "_")
    )

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

def reduce_to_stratifiable(df, label_col, n_splits):
    df = df.dropna(subset=[label_col]).copy()
    counts = df[label_col].astype(str).map(normalize_label).value_counts()
    bad = counts[counts < n_splits]
    if len(bad) > 0:
        raise ValueError(
            f"Cannot run StratifiedKFold(n_splits={n_splits}) because some classes in "
            f"'{label_col}' have fewer than {n_splits} examples: {bad.to_dict()}"
        )
    return df


# ============================================================
# TORCH DATASET
# ============================================================

class TextClassificationDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# METRICS
# ============================================================

def build_compute_metrics_fn(id2label):
    class_names = [id2label[i] for i in range(len(id2label))]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, preds)

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )

        metrics = {
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

        for i, cls in enumerate(class_names):
            key = cls.replace("-", "_")
            metrics[f"{key}_precision"] = precision[i]
            metrics[f"{key}_recall"] = recall[i]
            metrics[f"{key}_f1"] = f1[i]
            metrics[f"{key}_support"] = support[i]

        return metrics

    return compute_metrics


# ============================================================
# SINGLE FOLD
# ============================================================

def run_single_fold(
    train_df,
    test_df,
    text_col,
    topic_col,
    label_col,
    model_key,
    model_name,
    fold_idx,
    experiment_name,
    output_dir,
    max_length,
    num_epochs,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    weight_decay,
    val_size_within_train,
    early_stopping_patience,
    seed,
):
    train_sub_df, val_df = train_test_split(
        train_df,
        test_size=val_size_within_train,
        random_state=seed,
        stratify=train_df[label_col],
    )

    label_values = sorted(train_df[label_col].astype(str).map(normalize_label).unique().tolist())
    label2id = {lab: i for i, lab in enumerate(label_values)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def make_dataset(local_df):
        texts = local_df[text_col].astype(str).tolist()
        labels = [label2id[normalize_label(x)] for x in local_df[label_col].tolist()]
        return TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    train_dataset = make_dataset(train_sub_df)
    val_dataset = make_dataset(val_df)
    test_dataset = make_dataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_values),
        id2label=id2label,
        label2id=label2id,
    )

    run_name = f"{experiment_name}_{model_key}_fold{fold_idx}"
    trainer_output_dir = os.path.join(output_dir, f"tmp_{run_name}")
    
    training_args = make_training_args(
    output_dir=trainer_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=10,
    report_to="none",
    save_total_limit=1,
    seed=seed + fold_idx,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics_fn(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train()

    # Save training history
    history_df = pd.DataFrame(trainer.state.log_history)
    history_df.to_csv(
        os.path.join(output_dir, f"{run_name}_training_history.csv"),
        index=False,
    )

    # Predict on test fold
    #test_output = trainer.predict(test_dataset)
    #test_metrics = dict(test_output.metrics)
    
    test_output, avg_inference_time_sec, peak_gpu_memory_gb = measure_inference_time_and_memory(
    trainer, test_dataset)
    test_metrics = dict(test_output.metrics)    
 
    preds = np.argmax(test_output.predictions, axis=-1)
    gold = test_output.label_ids
    class_names = [id2label[i] for i in range(len(id2label))]

    report_dict = classification_report(
        gold,
        preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    pred_df = test_df[[text_col, topic_col, label_col]].copy().reset_index(drop=True)
    pred_df["gold_label"] = [id2label[i] for i in gold]
    pred_df["predicted_label"] = [id2label[i] for i in preds]
    pred_df["fold"] = fold_idx
    pred_df["model_key"] = model_key
    pred_df["model_name"] = model_name
    pred_df["experiment_name"] = experiment_name
    pred_df.to_csv(os.path.join(output_dir, f"{run_name}_predictions.csv"), index=False)

    with open(os.path.join(output_dir, f"{run_name}_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(os.path.join(output_dir, f"{run_name}_classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    row = {
        "experiment_name": experiment_name,
        "model_key": model_key,
        "model_name": model_name,
        "fold": fold_idx,
        "n_train": len(train_sub_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    row["avg_inference_time_sec"] = avg_inference_time_sec
    row["peak_gpu_memory_gb"] = peak_gpu_memory_gb
    
    for k, v in test_metrics.items():
        row[k.replace("test_", "")] = v

    for cls in class_names:
        key = cls.replace("-", "_")
        row[f"{key}_precision"] = report_dict[cls]["precision"]
        row[f"{key}_recall"] = report_dict[cls]["recall"]
        row[f"{key}_f1"] = report_dict[cls]["f1-score"]
        row[f"{key}_support"] = report_dict[cls]["support"]

    return row


# ============================================================
# CV EXPERIMENT
# ============================================================

def run_cv_experiment(
    df,
    text_col,
    topic_col,
    label_col,
    experiment_name,
    output_dir,
    models,
    n_splits,
    seed,
    max_length,
    num_epochs,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    weight_decay,
    val_size_within_train,
    early_stopping_patience,
):
    results = []

    y = df[label_col].astype(str).map(normalize_label)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for model_key, model_name in models.items():
        print(f"\n=== Running {experiment_name} | {model_key} ===")

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y), start=1):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            row = run_single_fold(
                train_df=train_df,
                test_df=test_df,
                text_col=text_col,
                topic_col=topic_col,
                label_col=label_col,
                model_key=model_key,
                model_name=model_name,
                fold_idx=fold_idx,
                experiment_name=experiment_name,
                output_dir=output_dir,
                max_length=max_length,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                weight_decay=weight_decay,
                val_size_within_train=val_size_within_train,
                early_stopping_patience=early_stopping_patience,
                seed=seed,
            )
            results.append(row)

    fold_df = pd.DataFrame(results)
    fold_df.to_csv(os.path.join(output_dir, f"{experiment_name}_all_folds_metrics.csv"), index=False)

    group_cols = ["experiment_name", "model_key", "model_name"]
    numeric_cols = [c for c in fold_df.columns if c not in group_cols + ["fold"]]

    summary_rows = []
    for keys, grp in fold_df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for c in numeric_cols:
            if pd.api.types.is_numeric_dtype(grp[c]):
                row[f"{c}_mean"] = grp[c].mean()
                row[f"{c}_std"] = grp[c].std(ddof=1) if len(grp) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, f"{experiment_name}_summary_metrics.csv"), index=False)

    return fold_df, summary_df


import inspect

def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("eval_strategy")
    else:
        raise TypeError("Neither eval_strategy nor evaluation_strategy is supported in this transformers version.")

    return TrainingArguments(**kwargs)



# ============================================================
# MAIN
# ============================================================
def measure_inference_time_and_memory(trainer, test_dataset):
    """
    Measure inference-only latency and peak GPU memory for the test dataset.
    Returns:
        test_output: trainer.predict(...) output
        avg_inference_time_sec: total prediction time / number of samples
        peak_gpu_memory_gb: peak allocated GPU memory during prediction, or None on CPU
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    test_output = trainer.predict(test_dataset)
    end = time.perf_counter()

    total_time = end - start
    n_samples = len(test_dataset)
    avg_inference_time_sec = total_time / n_samples if n_samples > 0 else None

    peak_gpu_memory_gb = None
    if torch.cuda.is_available():
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return test_output, avg_inference_time_sec, peak_gpu_memory_gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gold_standard_data_annotated.csv")
    parser.add_argument("--output-dir", default="finetune_ul_cv_outputs")
    parser.add_argument("--text-col", default="comment")
    parser.add_argument("--topic-col", default="topic")
    parser.add_argument("--label-col", default="final_label")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-size-within-train", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--run-pooled", action="store_true")
    parser.add_argument("--run-per-topic", action="store_true")
    args = parser.parse_args()

    if not args.run_pooled and not args.run_per_topic:
        raise ValueError("Set at least one of --run-pooled or --run-per-topic")

    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.data)
    require_columns(df, [args.text_col, args.topic_col, args.label_col])

    df = df.dropna(subset=[args.text_col, args.topic_col, args.label_col]).copy()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.topic_col] = df[args.topic_col].astype(str).str.strip().str.lower()
    df[args.label_col] = df[args.label_col].astype(str).map(normalize_label)

    with open(os.path.join(args.output_dir, f"run_config_{now_ts()}.json"), "w") as f:
        json.dump(
            {
                "data_path": args.data,
                "output_dir": args.output_dir,
                "models": DEFAULT_MODELS,
                "topics": args.topics,
                "n_splits": args.n_splits,
                "seed": args.seed,
                "run_pooled": args.run_pooled,
                "run_per_topic": args.run_per_topic,
                "text_col": args.text_col,
                "topic_col": args.topic_col,
                "label_col": args.label_col,
            },
            f,
            indent=2,
        )

    # Pooled UL
    if args.run_pooled:
        pooled_df = reduce_to_stratifiable(df.copy(), args.label_col, args.n_splits)

        print("\n" + "=" * 90)
        print("RUNNING POOLED UL ACROSS ALL TOPICS")
        print("=" * 90)

        run_cv_experiment(
            df=pooled_df[[args.text_col, args.topic_col, args.label_col]].copy(),
            text_col=args.text_col,
            topic_col=args.topic_col,
            label_col=args.label_col,
            experiment_name="ul_all_topics",
            output_dir=args.output_dir,
            models=DEFAULT_MODELS,
            n_splits=args.n_splits,
            seed=args.seed,
            max_length=args.max_length,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            weight_decay=args.weight_decay,
            val_size_within_train=args.val_size_within_train,
            early_stopping_patience=args.early_stopping_patience,
        )

    # UL per topic
    if args.run_per_topic:
        for topic in args.topics:
            topic_df = df[df[args.topic_col] == topic].copy()
            if topic_df.empty:
                print(f"[WARN] No rows found for topic: {topic}")
                continue

            topic_df = reduce_to_stratifiable(topic_df, args.label_col, args.n_splits)

            print("\n" + "=" * 90)
            print(f"RUNNING UL PER TOPIC: {topic}")
            print("=" * 90)

            run_cv_experiment(
                df=topic_df[[args.text_col, args.topic_col, args.label_col]].copy(),
                text_col=args.text_col,
                topic_col=args.topic_col,
                label_col=args.label_col,
                experiment_name=f"ul_{safe_name(topic)}",
                output_dir=args.output_dir,
                models=DEFAULT_MODELS,
                n_splits=args.n_splits,
                seed=args.seed,
                max_length=args.max_length,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                weight_decay=args.weight_decay,
                val_size_within_train=args.val_size_within_train,
                early_stopping_patience=args.early_stopping_patience,
            )

    print("\nDone.")
    print(f"All outputs saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
