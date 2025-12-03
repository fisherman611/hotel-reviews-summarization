import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "methods/hybrid_extractive_abstractive")
sys.path.append(PROJECT_ROOT / "utils")
import json

from aspect_classifier import AspectClassifier
from polarity_classifier import PolarityClassifier
from aspect_polarity_sentences_selector import AspectPolaritySentencesSelector
from aspect_abstractive_summarizer import AspectPolarityAbstractiveSummarizer
from utils.helpers import *
from tqdm.auto import tqdm

with open("methods/hybrid_extractive_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
ASPECTS = config["aspects"]
ASPECT_MODEL = config["aspect_model"]
THRESHOLD = config["threshold"]
POLARITIES = config["polarities"]
POLARITY_MODEL = config["polarity_model"]
K = config["k"]
SELECTOR_MODEL = config["selector_model"]
ABSTRACTIVE_MODEL = config["abstractive_model"]
MAX_NEW_TOKENS = config["max_new_tokens"]
DATA_PATH = config["data_path"]
SUMMARY_OUTPUT_PATH = "outputs/hybrid_extractive_abstractive_summaries.json"
os.makedirs("outputs", exist_ok=True)


aspect_abstractive_summarizer = AspectPolarityAbstractiveSummarizer(model_name=ABSTRACTIVE_MODEL, max_new_tokens=MAX_NEW_TOKENS)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    entities = json.load(f)
    
def run_hybrid_extractive_abstractive(
    grouped_output_path: str | None = None,
    topk_output_path: str | None = None,
    summary_output_path: str | None = None,
    aspect_threshold: float = THRESHOLD,
):
    """
    End-to-end pipeline:

    1. Run AspectClassifier per sentence -> sentence_aspects.
    2. Run PolarityClassifier per sentence - sentence_polarity
    2. Group sentences per aspect and polarity (just POSITIVE and NEGATIVE) -> reviews: {aspect: ["positive": sentences, "negative": sentences]}.
    3. Run AspectPolaritySentencesSelector -> top-K sentences per aspect and polarity.
    4. Build abstractive summaries per aspect.

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
    
    # ---------- Step 2: Polarity classification ----------
    polarity_clf = PolarityClassifier(
        model_name=POLARITY_MODEL, polarities=POLARITIES
    )
    print("STAGE 2: POLARITY CLASSIFICATION")
    for entity in tqdm(entities):
        for review in tqdm(entity.get("reviews", [])):
            sentences = review.get("sentences", [])
            sentence_polarity = []
            for sent in tqdm(sentences):
                pred = polarity_clf.predict(sent)
                sentence_polarity.append(pred["predicted_polarity"])
            review["sentence_polarity"] = sentence_polarity
            
    with open("data/polarity_aspect_classification.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=4)
        
    # ---------- Step 3: Group sentences by aspect and polarity ----------
    grouped_entities = group_sentences_by_aspect_polarity(
        entities,
        output_file=grouped_output_path
    )
        
    # ---------- Step 4: Aspect Polarity sentence selector ----------
    selector = AspectPolaritySentencesSelector(model_name=SELECTOR_MODEL, top_k=K)
    print("STAGE 3: ASPECT POLARITY SENTENCES SELECTOR")
    
    topk_entities = selector.process(grouped_entities)
    
    if topk_output_path is not None:
        with open(topk_output_path, "w", encoding="utf-8") as f:
            json.dump(topk_entities, f, ensure_ascii=False, indent=4)
    
    # ---------- Step 5: Aspect Abstractive Summarization ----------
    print("STAGE 4: ASPECT ABSTRACTIVE SUMMARIZATION")
    final_summaries = aspect_abstractive_summarizer.process(grouped_entities)

    if summary_output_path is not None:
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(final_summaries, f, ensure_ascii=False, indent=4)

    return {
        "grouped_entities": grouped_entities,
        "final_summaries": final_summaries,
    }
    
    
if __name__ == "__main__":
    result = run_hybrid_extractive_abstractive(
        summary_output_path=SUMMARY_OUTPUT_PATH,
        aspect_threshold=THRESHOLD,
    )
    
    print(result)
