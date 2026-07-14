# Misinformation Classification on Online Social Networks

## Scripts

**extract_data_from_reddit.py** — Extracts threads from Reddit given a CSV file of post hyperlinks. Saves them in JSON format.

**zero_shot_classification.py** — Runs zero shot classification on a labelled dataset and produces metrics.

**finetuning_colab_workflow.ipynb** — Used to run fine-tuning models on a labelled dataset. Run in Google Colab.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Direct dependencies (pinned versions from the development environment):

| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | 2.32.3 | HTTP requests (Reddit API) |
| `pandas` | 2.2.3 | Data manipulation |
| `praw` | 7.8.1 | Python Reddit API Wrapper |
| `scikit-learn` | 1.7.1 | Metrics and evaluation |
| `sentence-transformers` | 5.1.0 | Sentence embeddings |
| `transformers` | 4.55.2 | Pre-trained models (BART-MNLI, DistilBERT, RoBERTa) |
| `datasets` | 4.1.1 | HuggingFace dataset loading |
| `evaluate` | 0.4.6 | HuggingFace evaluation metrics |
| `seaborn` | 0.13.2 | Visualisation |
| `torch` | 2.8.0 | Deep learning backend |
| `accelerate` | 1.10.1 | Distributed training support |

> **Note:** The development environment used a CPU-only build of PyTorch (`torch==2.8.0+cpu`) on Windows. For GPU training (recommended for fine-tuning), install the appropriate CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

