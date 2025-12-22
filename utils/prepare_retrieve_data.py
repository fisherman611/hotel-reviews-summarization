import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "methods/hybrid_extractive_abstractive")
sys.path.append(PROJECT_ROOT / "utils")
import json

from methods.hybrid_extractive_abstractive.aspect_classifier import AspectClassifier
from methods.hybrid_extractive_abstractive.polarity_classifier import PolarityClassifier
from methods.hybrid_extractive_abstractive.aspect_polarity_sentences_selector import AspectPolaritySentencesSelector
from utils.helpers import *
from tqdm.auto import tqdm

with open("methods/hybrid_extractive_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    
ASPECTS = config["aspects"]
ASPECT_MODEL = config["aspect_model"]
THRESHOLD = config["threshold"]
POLARITIES = config["polarities"]
POLARITY_MODEL = config["polarity_model"]
K = 50
SELECTOR_MODEL = config["selector_model"]
DATA_PATH = Path("data/space_summ_val.json")
OUTPUT_PATH = Path(config["retrieve_data_path"])

os.makedirs(OUTPUT_PATH.parent, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    entities = json.load(f)
    
def run_prepare_retrieve_data(
    grouped_output_path: str | None = None,
    topk_output_path: str | None = None,
    aspect_threshold: float = THRESHOLD,
):
    """
    Pipeline to prepare data for retrieval (stops at top-k selection):

    1. Run AspectClassifier per sentence -> sentence_aspects.
    2. Run PolarityClassifier per sentence -> sentence_polarity.
    3. Group sentences per aspect and polarity (just POSITIVE and NEGATIVE) -> reviews: {aspect: ["positive": sentences, "negative": sentences]}.
    4. Run AspectPolaritySentencesSelector -> top-K sentences per aspect and polarity.

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
        for review in entity.get("reviews", []):
            sentences = review.get("sentences", [])
            sentence_aspects = []
            for sent in sentences:
                pred = aspect_clf.predict(sent)
                sentence_aspects.append(pred["predicted_aspects"])
            review["sentence_aspects"] = sentence_aspects
    
    # ---------- Step 2: Polarity classification ----------
    polarity_clf = PolarityClassifier(
        model_name=POLARITY_MODEL, polarities=POLARITIES
    )
    print("STAGE 2: POLARITY CLASSIFICATION")
    for entity in tqdm(entities):
        for review in entity.get("reviews", []):
            sentences = review.get("sentences", [])
            sentence_polarity = []
            for sent in sentences:
                pred = polarity_clf.predict(sent)
                sentence_polarity.append(pred["predicted_polarity"])
            review["sentence_polarity"] = sentence_polarity
        
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
    
if __name__ == "__main__":
    result = run_prepare_retrieve_data(
        topk_output_path=OUTPUT_PATH,
        aspect_threshold=THRESHOLD,
    )
