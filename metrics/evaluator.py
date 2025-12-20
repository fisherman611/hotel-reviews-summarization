import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.metrics import compute_all_metrics_single
import json
from tqdm.auto import tqdm

# INPUT_PATH = Path("outputs/baseline/baseline_deberta_summaries.json")
INPUT_PATH = Path("outputs/baseline/baseline_bart_summaries.json")

# RESULT_PATH = Path("results/automatic_eval/baseline/baseline_deberta.json")
RESULT_PATH = Path("results/automatic_eval/baseline/baseline_bart.json")
os.makedirs(RESULT_PATH.parent, exist_ok=True)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
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
        "bertscore_f1": [],
    },
    "location": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    },
    "service": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    },
    "cleanliness": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    },
    "building": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    },
    "food": {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    },
}

for i in tqdm(range(len(output))):
    sample = output[i]
    generated_summaries = sample["generated_summaries"]
    golden_summaries = sample["golden_summaries"]
    for aspect in aspects:
        pred = generated_summaries[aspect]
        gold_refs = golden_summaries[aspect]
        scores = compute_all_metrics_single(pred, gold_refs, lang="en")
        for key, value in scores.items():
            final_scores[aspect][key].append(value)

for par_key, value_dictionary in final_scores.items():
    for key, value in value_dictionary.items():
        final_scores[par_key][key] = sum(value) / len(value)

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_scores, f, ensure_ascii=False, indent=2)

for par_key, value_dictionary in final_scores.items():
    print(f"Aspect: {par_key}")
    for key, value in value_dictionary.items():
        print(f"{key}: {value}")
    print("=" * 80)
