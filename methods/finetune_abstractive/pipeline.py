import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "methods/finetune_abstractive")
import json

from aspect_abstractive_summarizer import AspectAbstractiveSummarizer
from tqdm.auto import tqdm

with open("methods/finetune_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ASPECTS = config["aspects"]
ABSTRACTIVE_MODEL = config["abstractive_model"]
MAX_NEW_TOKENS = config["max_new_tokens"]
DATA_PATH = config["data_path"]
SUMMARY_OUTPUT_PATH = "outputs/finetune_abstractive_summaries.json"
os.makedirs("outputs", exist_ok=True)


def run_finetune_abstractive_pipeline(
    data_path: str = DATA_PATH,
    summary_output_path: str = SUMMARY_OUTPUT_PATH,
    model_name: str = ABSTRACTIVE_MODEL,
    max_new_tokens: int = MAX_NEW_TOKENS,
    template_index: int = 0,
    aspects: list = None,
):
    """
    The finetuned model directly generates
    aspect-specific summaries based on instruction prompts (FLAN approach).

    Pipeline:
    1. Load entities with reviews (original format)
    2. For each entity and aspect, generate summary using instruction prompt
    3. Save results

    Input format:
        {
            "entity_id": ...,
            "entity_name": ...,
            "reviews": [
                {
                    "review_id": ...,
                    "sentences": [...],
                    "rating": ...
                },
                ...
            ],
            "summaries": { aspect: [golden summaries...] }  # Optional
        }

    Args:
        data_path: Path to input JSON file
        summary_output_path: Path to save output
        model_name: Finetuned model name/path
        max_new_tokens: Max tokens to generate
        template_index: Which instruction template to use (0-9)
        aspects: List of aspects to summarize
    """
    if aspects is None:
        aspects = ASPECTS

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    # Initialize summarizer with finetuned model
    print(f"Loading finetuned model: {model_name}")
    summarizer = AspectAbstractiveSummarizer(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        template_index=template_index,
    )

    # Generate summaries directly from reviews
    print("Generating aspect summaries using instruction prompts...")
    final_summaries = []

    for entity in tqdm(entities, desc="Processing entities"):
        result = summarizer.summarize_entity(entity, aspects=aspects)
        final_summaries.append(result)

    # Save results
    if summary_output_path is not None:
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(final_summaries, f, ensure_ascii=False, indent=4)
        print(f"Saved summaries to: {summary_output_path}")

    return final_summaries


if __name__ == "__main__":
    # Run pipeline with finetuned model
    results = run_finetune_abstractive_pipeline(
        data_path=DATA_PATH,
        summary_output_path=SUMMARY_OUTPUT_PATH,
        model_name=ABSTRACTIVE_MODEL,  # Replace with your finetuned model path
        template_index=0,  # Use first template (same as used in training)
        aspects=ASPECTS,
    )

    print(f"\nProcessed {len(results)} entities")
    print(f"Results saved to: {SUMMARY_OUTPUT_PATH}")
