import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import pipeline

""" Import configuration for zero-shot classification """
# from configs.ws_config import ZS_MODEL_NAME, OUTPUT_METRICS, LABEL_SET, GOLD_STANDARD_TOPIC # ENVIRONMENT TOPIC
# from configs.tb_config import ZS_MODEL_NAME, OUTPUT_METRICS, LABEL_SET, GOLD_STANDARD_TOPIC # HEALTH TOPIC
from configs.it_config import ZS_MODEL_NAME, OUTPUT_METRICS, LABEL_SET, GOLD_STANDARD_TOPIC # IMMIGRATION TOPIC

def init_classifier(model_name: str):
    """
    Initialize a Hugging Face zero-shot classifier using the specified model.
    
    Args:
        model_name (str): The Hugging Face model name.
        
    Returns:
        pipeline object: Hugging Face zero-shot-classification pipeline.
    """
    return pipeline("zero-shot-classification", model=model_name)


def load_and_prepare_data(csv_path: str, label_mapping_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load gold standard data and label mapping and filter for a specific topic.
    
    Args:
        csv_path (str): Path to the CSV file containing gold standard data.
        label_mapping_path (str): Path to JSON file with label mapping.
        
    Returns:
        tuple: Filtered DataFrame and label mapping dictionary.
    """
    df = pd.read_csv(csv_path)
    
    # Load label mapping
    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)
        
    # Filter for specific topic
    df = df[df["topic"] == GOLD_STANDARD_TOPIC]
    
    # Normalize labels to lowercase
    df["assigned_label"] = df["assigned_label"].str.lower()
    
    return df, label_mapping



def is_valid_comment(comment: str, min_words: int = 3) -> bool:
    """
    Determine if a comment is valid for classification.
    
    Args:
        comment (str): Comment text.
        min_words (int): Minimum word count to consider comment valid.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if not comment or comment.lower() in {"[removed]", "[deleted]"}:
        return False
    if len(comment.split()) < min_words:
        return False
    return True


def classify_comments(data: pd.DataFrame, labels: list, classifier, label_mapping: dict) -> pd.DataFrame:
    """
    Perform zero-shot classification on each comment in the dataset.
    
    Args:
        data (pd.DataFrame): DataFrame with comments to classify.
        labels (list): Candidate labels for classification.
        classifier: Hugging Face zero-shot-classification pipeline.
        label_mapping (dict): Mapping of predicted labels to standardised labels.
    
    Returns:
        pd.DataFrame: DataFrame with original and predicted labels, mapped labels, and scores.
    """
    results = []
    
    for idx, row in data.iterrows():
        comment = row['comment']
        
        base_result = {
            "id": row['id'],
            "comment": comment,
            "assigned_label": row['assigned_label'],
        }
        
        # Skip invalid comments
        if not is_valid_comment(comment):
            pred_label = "other"
            results.append({
                **base_result,
                "pred_label": pred_label,
                "pred_label_mapped": label_mapping.get(pred_label, pred_label),
                "pred_score": 1.0,
            })
            continue
        
        # Run classifier on valid comments
        result = classifier(comment, candidate_labels=labels, multi_label=False)
        pred_label = result["labels"][0]
        results.append({
            **base_result,
            "pred_label": pred_label,
            "pred_label_mapped": label_mapping.get(pred_label, pred_label),
            "pred_score": result["scores"][0],
        })
    
    return pd.DataFrame(results)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, metric_name: str = "") -> dict:
    """
    Compute classification metrics including accuracy, confusion matrix, per-class metrics,
    and macro/weighted averages.
    
    Args:
        y_true (pd.Series): Ground truth labels.
        y_pred (pd.Series): Predicted labels.
        metric_name (str): Optional metric name for labeling outputs.
        
    Returns:
        dict: Dictionary containing all computed metrics.
    """
    labels = sorted(y_true.unique())
    
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'True Label'
    cm_df.columns.name = 'Predicted Label'
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    
    per_class_metrics = pd.DataFrame({
        'Label': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    overall_metrics = {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1_score": f1_macro
        },
        "weighted_avg": {
            "precision": precision_weighted,
            "recall": recall_weighted,
            "f1_score": f1_weighted
        }
    }
    
    return {
        'metric_name': metric_name,
        'accuracy': accuracy,
        'confusion_matrix': cm_df,
        'classification_report': report,
        'per_class_metrics': per_class_metrics,
        'overall_metrics': overall_metrics
    }



def save_metrics(metrics: dict, output_dir: Path, prefix: str = ""):
    """
    Save classification metrics and visualizations to files.
    
    Args:
        metrics (dict): Metrics dictionary from calculate_metrics().
        output_dir (Path): Directory to save outputs.
        prefix (str): Optional prefix for filenames.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    prefix = f"{prefix}_" if prefix else ""
    
    # Save classification report JSON and CSV
    with open(output_dir / f"{prefix}classification_report.json", "w") as f:
        json.dump(metrics['classification_report'], f, indent=4)
    
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_df.to_csv(output_dir / f"{prefix}classification_report.csv")
    
    # Save confusion matrix CSV
    metrics['confusion_matrix'].to_csv(output_dir / f"{prefix}confusion_matrix.csv")
    
    # Save per-class metrics CSV
    metrics['per_class_metrics'].to_csv(output_dir / f"{prefix}per_class_metrics.csv", index=False)
    
    # Save overall metrics JSON
    with open(output_dir / f"{prefix}overall_metrics.json", "w") as f:
        json.dump(metrics['overall_metrics'], f, indent=4)
    
    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'})
    title = f"Confusion Matrix - {metrics['metric_name']}" if metrics['metric_name'] else "Confusion Matrix"
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()



def main():
    """
    Main execution function:
    - Load data and label mapping
    - Initialize zero-shot classifier
    - Classify comments
    - Calculate, print, and save metrics
    """
    
    # Load and prepare data
    gold_standard_csv = Path("input/gold_standard_data.csv")
    gold_standard_df, label_mapping = load_and_prepare_data(
        gold_standard_csv, 
        "configs/label_mapping.json"
    )
    
    # Initialise classifier
    classifier = init_classifier(ZS_MODEL_NAME)
    
    # Classify comments
    results_df = classify_comments(gold_standard_df, LABEL_SET, classifier, label_mapping)
    
    # Save raw classification results
    results_df.to_csv(OUTPUT_METRICS / "classification_results.csv", index=False)
    print(f"\n✓ Classification results saved to '{OUTPUT_METRICS / 'classification_results.csv'}'")
    
    
    # Calculate metrics for mapped labels
    print("\nCalculating mapped metrics...")
    mapped_metrics = calculate_metrics(
        results_df['assigned_label'], 
        results_df['pred_label_mapped'],
        metric_name="Mapped Labels"
    )
        
    # Save mapped metrics
    save_metrics(mapped_metrics, OUTPUT_METRICS, prefix="mapped")


if __name__ == "__main__":
    main()