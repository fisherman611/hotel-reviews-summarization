import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "methods/finetune_abstractive")
sys.path.append(PROJECT_ROOT / "utils")
import json

from aspect_classifier import AspectClassifier
from aspect_abstractive_summarizer import AspectAbstractiveSummarizer
from utils.helpers import group_sentences_by_aspect
from tqdm.auto import tqdm

with open("methods/finetune_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
ASPECTS = config["aspects"]
ASPECT_MODEL = config["aspect_model"]
THRESHOLD = config["threshold"]
ABSTRACTIVE_MODEL = config["abstractive_model"]
MAX_NEW_TOKENS = config["max_new_tokens"]
DATA_PATH = config["data_path"]
SUMMARY_OUTPUT_PATH = "outputs/finetune_abstractive_summaries.json"
os.makedirs("outputs", exist_ok=True)

aspect_abstractive_summarizer = AspectAbstractiveSummarizer(model_name=ABSTRACTIVE_MODEL, max_new_tokens=MAX_NEW_TOKENS)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    entities = json.load(f)

def run_finetune_abstractive_pipeline(
    grouped_output_path: str | None = None,
    summary_output_path: str | None = None,
    aspect_threshold: float = THRESHOLD,
):
    """
    End-to-end pipeline:

    1. Run AspectClassifier per sentence -> sentence_aspects.
    2. Group sentences per aspect -> reviews: {aspect: [sentences]}.
    3. Build abstractive summaries per aspect.

    Know that each entity in input has:
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
            "summaries": { aspect: [golden summaries...] }
        }
    """
    # ---------- Step 1: Aspect classification ----------
    aspect_clf = AspectClassifier(
        model_name=ASPECT_MODEL, aspects=ASPECTS, threshold=aspect_threshold
    )
    print("STAGE 1: ASPECT CLASSIFICATION")
    for entity in tqdm(entities):
        for review in tqdm(entity.get("reviews", [])):
            sentences = review.get("sentences", [])
            sentence_aspects = []
            for sent in tqdm(sentences):
                pred = aspect_clf.predict(sent)
                sentence_aspects.append(pred["predicted_aspects"])
            review["sentence_aspects"] = sentence_aspects
    
    # ---------- Step 2: Group sentences by aspect ----------
    grouped_entities = group_sentences_by_aspect(
        entities,
        output_file=grouped_output_path,
    )
    
     # ---------- Step 3: Aspect ABSTRACTIVE summaries ----------
    print("STAGE 2: ASPECT ABSTRACTIVE SUMMARIZATION")
    final_summaries = aspect_abstractive_summarizer.process(grouped_entities)

    if summary_output_path is not None:
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(final_summaries, f, ensure_ascii=False, indent=4)

    return {
        "grouped_entities": grouped_entities,
        "final_summaries": final_summaries,
    }

if __name__ == "__main__":
    # Example usage

    result = run_finetune_abstractive_pipeline(
        summary_output_path=SUMMARY_OUTPUT_PATH,
        aspect_threshold=THRESHOLD,
    )
    print(result)