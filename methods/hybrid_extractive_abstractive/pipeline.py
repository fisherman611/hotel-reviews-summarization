import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(PROJECT_ROOT / "methods/hybrid_extractive_abstractive")
sys.path.append(PROJECT_ROOT / "utils")
import json
from typing import Dict, List, Any, Optional

from aspect_classifier import AspectClassifier
from polarity_classifier import PolarityClassifier
from aspect_polarity_sentences_selector import AspectPolaritySentencesSelector
from aspect_abstractive_summarizer import AspectPolarityAbstractiveSummarizer
from retrieve_similar_examples import HybridRetriever, load_retriever, build_retrieval_index
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
NUM_EXAMPLES = config["num_examples"]
RETRIEVAL_THRESHOLD = config.get("retrieval_threshold", 0.5)
USE_RETRIEVAL = True if config["use_retrieval"] else False
DATA_PATH = Path(config["data_path"])
RETRIEVE_DATA_PATH = Path(config["retrieve_data_path"])
INDEX_DIR = Path(config["index_dir"])
SUMMARY_OUTPUT_PATH = Path(config["summary_output_path"])
GROUPED_OUTPUT_PATH = Path(config["grouped_output_path"])
TOPK_OUTPUT_PATH = Path(config["topk_output_path"])
POLARITY_ASPECT_PATH = Path(config["polarity_aspect_path"])

os.makedirs(SUMMARY_OUTPUT_PATH.parent, exist_ok=True)
os.makedirs(GROUPED_OUTPUT_PATH.parent, exist_ok=True)
os.makedirs(TOPK_OUTPUT_PATH.parent, exist_ok=True)
os.makedirs(POLARITY_ASPECT_PATH.parent, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

aspect_abstractive_summarizer = AspectPolarityAbstractiveSummarizer(model_name=ABSTRACTIVE_MODEL, max_new_tokens=MAX_NEW_TOKENS)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    entities = json.load(f)
    
def run_hybrid_extractive_abstractive(
    grouped_output_path: str | None = None,
    topk_output_path: str | None = None,
    summary_output_path: str | None = None,
    aspect_threshold: float = THRESHOLD,
    use_retrieval: bool = True,
    num_examples: int = NUM_EXAMPLES,
    retrieval_threshold: float = RETRIEVAL_THRESHOLD,
):
    """
    End-to-end pipeline:

    1. Run AspectClassifier per sentence -> sentence_aspects.
    2. Run PolarityClassifier per sentence - sentence_polarity
    2. Group sentences per aspect and polarity (just POSITIVE and NEGATIVE) -> reviews: {aspect: ["positive": sentences, "negative": sentences]}.
    3. Run AspectPolaritySentencesSelector -> top-K sentences per aspect and polarity.
    4. Retrieve similar examples for few-shot prompting (optional).
    5. Build abstractive summaries per aspect with few-shot examples (or zero-shot if no examples).

    Args:
        grouped_output_path: Path to save grouped entities
        topk_output_path: Path to save top-k sentences
        summary_output_path: Path to save final summaries
        aspect_threshold: Threshold for aspect classification
        use_retrieval: Whether to use retrieval for few-shot examples
        num_examples: Number of examples to retrieve
        retrieval_threshold: Minimum hybrid score threshold for retrieval (default: 0.5)

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
            
    with open(POLARITY_ASPECT_PATH, "w", encoding="utf-8") as f:
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
    
    # if topk_output_path is not None:
    #     with open(topk_output_path, "w", encoding="utf-8") as f:
    #         json.dump(topk_entities, f, ensure_ascii=False, indent=4)
    
    # ---------- Step 5: Retrieve similar examples for few-shot prompting ----------
    entity_examples: Dict[str, List[Dict[str, Any]]] = {}
    
    if use_retrieval:
        print("STAGE 4: RETRIEVING SIMILAR EXAMPLES FOR FEW-SHOT PROMPTING")
        
        # Load or build the retrieval index
        if INDEX_DIR.exists() and (INDEX_DIR / "faiss_index.bin").exists():
            print(f"Loading existing retrieval index from {INDEX_DIR}...")
            retriever = load_retriever(INDEX_DIR)
        else:
            print(f"Building new retrieval index from {RETRIEVE_DATA_PATH}...")
            retriever = build_retrieval_index(
                retrieve_data_path=RETRIEVE_DATA_PATH,
                index_dir=INDEX_DIR,
            )
        
        # Retrieve similar examples for each entity
        zero_shot_count = 0
        for entity in tqdm(entities, desc="Retrieving examples"):
            entity_id = entity["entity_id"]
            
            # Retrieve similar examples (excluding the entity itself)
            retrieved = retriever.retrieve(
                query_entity=entity,
                top_k=num_examples,
                exclude_entity_ids=[entity_id],
                threshold=retrieval_threshold,
            )
            
            # Convert RetrievedExample objects to dicts for the summarizer
            examples_for_entity = []
            for ex in retrieved:
                examples_for_entity.append({
                    "entity_id": ex.entity_id,
                    "entity_name": ex.entity_name,
                    "topk_sentences": ex.topk_sentences,
                    "summaries": ex.summaries,
                })
            
            entity_examples[entity_id] = examples_for_entity
            
            if len(examples_for_entity) == 0:
                zero_shot_count += 1
                print(f"  Entity '{entity.get('entity_name', entity_id)}': No examples retrieved (below threshold {retrieval_threshold}). Will use zero-shot.")
            else:
                print(f"  Entity '{entity.get('entity_name', entity_id)}': retrieved {len(examples_for_entity)} examples")
        
        print(f"\nRetrieval Summary:")
        print(f"  Total entities: {len(entities)}")
        print(f"  Zero-shot (no examples): {zero_shot_count}")
        print(f"  Few-shot (with examples): {len(entities) - zero_shot_count}")
    else:
        print("STAGE 4: RETRIEVAL DISABLED - USING ZERO-SHOT PROMPTING FOR ALL ENTITIES")
        print(f"  All {len(entities)} entities will use zero-shot prompting.")
    
    # ---------- Step 6: Aspect Abstractive Summarization ----------
    if use_retrieval and entity_examples:
        print("STAGE 5: ASPECT ABSTRACTIVE SUMMARIZATION (FEW-SHOT/ZERO-SHOT HYBRID)")
    else:
        print("STAGE 5: ASPECT ABSTRACTIVE SUMMARIZATION (ZERO-SHOT)")
    
    final_summaries = aspect_abstractive_summarizer.process(
        topk_entities, 
        entity_examples=entity_examples if use_retrieval else None
    )

    if summary_output_path is not None:
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(final_summaries, f, ensure_ascii=False, indent=4)

    return {
        "topk_entities": topk_entities,
        "final_summaries": final_summaries,
    }
    
    
if __name__ == "__main__":
    result = run_hybrid_extractive_abstractive(
        # grouped_output_path=GROUPED_OUTPUT_PATH,
        summary_output_path=SUMMARY_OUTPUT_PATH,
        # aspect_threshold=THRESHOLD,
        # topk_output_path=TOPK_OUTPUT_PATH,
        use_retrieval=USE_RETRIEVAL,
        num_examples=NUM_EXAMPLES,
        # retrieval_threshold=RETRIEVAL_THRESHOLD,  # Uses config value by default (0.5)
    )