import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer, util
import torch
import json
from typing import Dict, List, Any

with open("methods/hybrid_extractive_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

SELECTOR_MODEL = config["selector_model"]


class AspectSentencesSelector:
    def __init__(
        self, model_name: str = SELECTOR_MODEL, top_k: int = 3, device: str = None
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_list(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return None
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

    def rank_sentences_by_context(
        self, sentences: List[str], context_sentences: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank sentences based on their similarity to the centroid of context embeddings.
        This selects the most representative sentences from the context.
        This approach works for both training and inference since it doesn't require golden summaries.
        
        Args:
            sentences: Sentences to rank
            context_sentences: Optional broader context to compute centroid from.
                              If None, uses the sentences themselves as context.
        """
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [{"sentence": sentences[0], "score": 1.0}]

        # Use provided context or fall back to sentences themselves
        context = context_sentences if context_sentences else sentences
        
        # Encode sentences to rank and context
        sentence_embs = self.encode_list(sentences)
        context_embs = self.encode_list(context)
        
        # Calculate the centroid (mean) of all context embeddings
        centroid = context_embs.mean(dim=0, keepdim=True)
        
        # Calculate similarity of each sentence to the centroid
        sims = util.cos_sim(sentence_embs, centroid)
        scores = sims.squeeze()
        
        # Create ranked list
        ranked = []
        for sent, score in zip(sentences, scores):
            ranked.append({"sentence": sent, "score": float(score.item())})
        
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def get_top_k(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select top-k sentences for each aspect using context-based ranking.
        Uses centroid similarity without requiring golden summaries.
        """
        aspect_to_sentences = entity.get("grouped_reviews", {})

        topk_sentences = {}
        aspect_sentence_scores = {}

        for aspect, sentences in aspect_to_sentences.items():
            # Use context-based ranking with the aspect's sentences as context
            ranked = self.rank_sentences_by_context(sentences)
            topk = ranked[: self.top_k]
            topk_sentences[aspect] = [item["sentence"] for item in topk]
            aspect_sentence_scores[aspect] = ranked

        return {
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "topk_sentences": topk_sentences,
            "aspect_sentence_scores": aspect_sentence_scores,
        }

    def process(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.get_top_k(e) for e in entities]

class AspectPolaritySentencesSelector(AspectSentencesSelector):
    def get_top_k(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select top-k sentences for each aspect-polarity combination.
        Uses context-based ranking with full review embeddings as the anchor.
        
        This approach:
        1. Joins sentences in each review to create full reviews (e.g., 100 reviews)
        2. Embeds all 100 reviews → 100 context embeddings
        3. Calculates mean of embeddings → Centroid of context
        4. Ranks aspect-polarity sentences against this centroid
        """
        aspect_to_polarity_sentences = entity.get("grouped_reviews", {})
        
        # Build full review context from original reviews
        # Extract reviews and join sentences to create full review texts
        reviews = entity.get("reviews", [])
        full_reviews = []
        for review in reviews:
            sentences = review.get("sentences", [])
            # Join all sentences in the review into a single text
            full_review = " ".join(sentences)
            full_reviews.append(full_review)
        
        # Use full reviews as context for embedding
        # This creates embeddings for each of the 100 reviews
        review_context = full_reviews if full_reviews else []

        topk_sentences = {}
        aspect_sentence_scores = {}

        for aspect, polarity_dict in aspect_to_polarity_sentences.items():
            topk_sentences[aspect] = {}
            aspect_sentence_scores[aspect] = {}

            # polarity_dict is like {"negative": [...], "positive": [...]}
            for polarity, sentences in polarity_dict.items():
                ranked = self.rank_sentences_by_context(sentences, review_context)
                topk = ranked[: self.top_k]

                topk_sentences[aspect][polarity] = [item["sentence"] for item in topk]
                aspect_sentence_scores[aspect][polarity] = ranked

        return {
            "entity_id": entity["entity_id"],
            "entity_name": entity.get("entity_name"),
            "topk_sentences": topk_sentences,
            "aspect_sentence_scores": aspect_sentence_scores,
        }
