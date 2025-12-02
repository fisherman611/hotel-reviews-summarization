import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "methods/baseline")
sys.path.append(PROJECT_ROOT / "utils")
import json

from aspect_classifier import AspectClassifier
from aspect_sentences_selector import AspectSentencesSelector
from aspect_extractive_summarizer import aspect_extractive_summarizer
from utils.helpers import group_sentences_by_aspect
from tqdm.auto import tqdm

with open("methods/baseline/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ASPECTS = config["aspects"]
ASPECT_MODEL = config["aspect_model"]
THRESHOLD = config["threshold"]
SELECTOR_MODEL = config["selector_model"]
K = config["k"]
DATA_PATH = config["data_path"]
SUMMARY_OUTPUT_PATH = "outputs/baseline_summaries.json"
os.makedirs("outputs", exist_ok=True)

# ---------- Step 1: Load input ----------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    entities = json.load(f)


def run_baseline_pipeline(
    grouped_output_path: str | None = None,
    topk_output_path: str | None = None,
    summary_output_path: str | None = None,
    aspect_threshold: float = 0.5,
):
    """
    End-to-end pipeline:

    1. Run AspectClassifier per sentence -> sentence_aspects.
    2. Group sentences per aspect -> reviews: {aspect: [sentences]}.
    3. Run AspectSentencesSelector -> top-K sentences per aspect.
    4. Build extractive summaries per aspect.

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
    # NOTE: group_sentences_by_aspect already carries over "summaries" if present.

    # ---------- Step 4: Aspect sentence selector ----------
    selector = AspectSentencesSelector(model_name=SELECTOR_MODEL, top_k=K)
    print("STAGE 2: ASPECT SENTENCES SELECTOR")

    topk_entities = selector.process(grouped_entities)

    if topk_output_path is not None:
        with open(topk_output_path, "w", encoding="utf-8") as f:
            json.dump(topk_entities, f, ensure_ascii=False, indent=4)

    # ---------- Step 5: Aspect extractive summaries ----------
    print("STAGE 3: ASPECT EXTRACTIVE SUMMARIZATION")
    final_summaries = []
    for ent in topk_entities:
        topk_sentences = ent["topk_sentences"]
        aspect_summaries = aspect_extractive_summarizer(topk_sentences)

        final_summaries.append(
            {
                "entity_id": ent["entity_id"],
                "entity_name": ent["entity_name"],
                "aspect_summaries": aspect_summaries,
            }
        )

    if summary_output_path is not None:
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(final_summaries, f, ensure_ascii=False, indent=4)

    return {
        "grouped_entities": grouped_entities,
        "topk_entities": topk_entities,
        "final_summaries": final_summaries,
    }


if __name__ == "__main__":
    # Example usage

    result = run_baseline_pipeline(
        summary_output_path=SUMMARY_OUTPUT_PATH,
        aspect_threshold=THRESHOLD,
    )
    print(result)
