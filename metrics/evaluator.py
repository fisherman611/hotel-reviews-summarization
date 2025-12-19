import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from metrics.metrics import compute_all_metrics_single
import json

with open("outputs/baseline_summaries.json", "r", encoding="utf-8") as f:
# with open("outputs/finetune_abstractive_summaries.json", "r", encoding="utf-8") as f:
# with open("outputs/hybrid_extractive_abstractive_summaries.json", "r", encoding="utf-8") as f:
    output = json.load(f)

aspects = ["rooms", "location", "service", "cleanliness", "building", "food"]
final_scores = {
    "rooms": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": []
    },
    "location": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": []
    },
    "service": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": []
    },
    "cleanliness": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": []
    },
    "building": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": []
    },
    "food": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": []
    },
    
}

for i in range(len(output)):
    sample = output[i]
    generated_summaries = sample["generated_summaries"]
    golden_summaries = sample["golden_summaries"]
    for aspect in aspects:
        pred = generated_summaries[aspect]
        gold_refs = golden_summaries[aspect]
        scores = compute_all_metrics_single(pred, gold_refs, lang='en')
        for key, value in scores.items():
            final_scores[aspect][key].append(value)
print(final_scores)
    