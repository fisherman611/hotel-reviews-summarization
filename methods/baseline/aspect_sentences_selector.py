import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import json
from typing import Dict, List, Any


class AspectSentencesSelector:
    def __init__(self, top_k: int = 3, seed: int = 42) -> None:
        """
        Random sentence selector for baseline method.
        
        Args:
            top_k: Number of sentences to select per aspect
            seed: Random seed for reproducibility
        """
        self.top_k = top_k
        self.seed = seed
        random.seed(self.seed)

    def select_random_sentences(
        self, sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Randomly select sentences (baseline approach).
        
        Args:
            sentences: List of sentences to select from
            
        Returns:
            List of dictionaries with sentence and score (random)
        """
        if not sentences:
            return []

        # Create a list with random scores
        ranked = []
        for sent in sentences:
            ranked.append({"sentence": sent, "score": random.random()})

        # Sort by random score
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def get_top_k(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select top-k sentences randomly for each aspect.
        
        Args:
            entity: Dictionary containing entity_id, reviews, and summaries
            
        Returns:
            Dictionary with entity_id, topk_sentences, and aspect_sentence_scores
        """
        aspect_to_sentences = entity.get("grouped_reviews", {})

        topk_sentences = {}
        aspect_sentence_scores = {}

        for aspect, sentences in aspect_to_sentences.items():
            # Random selection (baseline approach)
            ranked = self.select_random_sentences(sentences)
            topk = ranked[: self.top_k]
            topk_sentences[aspect] = [item["sentence"] for item in topk]
            aspect_sentence_scores[aspect] = ranked

        return {
            "entity_id": entity["entity_id"],
            "topk_sentences": topk_sentences,
            "aspect_sentence_scores": aspect_sentence_scores,
        }

    def process(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.get_top_k(e) for e in entities]
