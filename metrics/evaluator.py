import os
import sys
import json
import argparse
from pathlib import Path
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.metrics import compute_all_metrics_single

def main():
    parser = argparse.ArgumentParser(description="Evaluate hotel review summaries using standard metrics (ROUGE, BLEU, etc.)")
    parser.add_argument("--input_file", required=True, help="Path to JSON file with summaries to evaluate")
    parser.add_argument("-o", "--output", required=True, help="Path to save evaluation results")
    args = parser.parse_args()

    # INPUT_PATH = Path("outputs/baseline/baseline_deberta_summaries.json")
    # INPUT_PATH = Path("outputs/baseline/baseline_bart_summaries.json")

    # RESULT_PATH = Path("results/automatic_eval/baseline/baseline_deberta.json")
    # RESULT_PATH = Path("results/automatic_eval/baseline/baseline_bart.json")

    INPUT_PATH = Path(args.input_file)
    RESULT_PATH = Path(args.output)
    
    if not INPUT_PATH.exists():
        print(f"Error: Input file {INPUT_PATH} does not exist.")
        sys.exit(1)

    os.makedirs(RESULT_PATH.parent, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    aspects = ["rooms", "location", "service", "cleanliness", "building", "food"]
    
    # Initialize dictionary for storing individual scores per aspect
    final_scores = {
        aspect: {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bleu": [],
            "meteor": [],
            "bertscore_precision": [],
            "bertscore_recall": [],
            "bertscore_f1": [],
        } for aspect in aspects
    }

    print(f"Evaluating {len(output_data)} samples...")
    for i in tqdm(range(len(output_data))):
        sample = output_data[i]
        generated_summaries = sample["generated_summaries"]
        golden_summaries = sample["golden_summaries"]
        for aspect in aspects:
            pred = generated_summaries[aspect]
            gold_refs = golden_summaries[aspect]
            scores = compute_all_metrics_single(pred, gold_refs, lang="en")
            for key, value in scores.items():
                final_scores[aspect][key].append(value)

    # Compute averages
    for par_key, value_dictionary in final_scores.items():
        for key, value in value_dictionary.items():
            if value:
                final_scores[par_key][key] = sum(value) / len(value)
            else:
                final_scores[par_key][key] = 0.0

    # Save results
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_scores, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 40 + " EVALUATION SUMMARY " + "=" * 40)
    for par_key, value_dictionary in final_scores.items():
        print(f"Aspect: {par_key}")
        for key, value in value_dictionary.items():
            print(f"  {key:20}: {value:.4f}")
        print("-" * 100)
    print(f"Results saved to: {RESULT_PATH}")

if __name__ == "__main__":
    main()
