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

with open("methods/baseline/config.json", "r", encoding="utf-8") as f:
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
        )

    def rank_sentences_for_aspect(
        self, sentences: List[str], summaries: List[str]
    ) -> List[Dict[str, Any]]:
        if not sentences or not summaries:
            return []

        summary_embs = self.encode_list(summaries)
        sentence_embs = self.encode_list(sentences)

        sims = util.cos_sim(sentence_embs, summary_embs)
        avg_scores = sims.mean(dim=1)

        ranked = []
        for sent, score in zip(sentences, avg_scores):
            ranked.append({"sentence": sent, "score": float(score.item())})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def get_top_k(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        aspect_to_sentences = entity.get("reviews", {})
        aspect_to_summaries = entity.get("summaries", {})

        topk_sentences = {}
        aspect_sentence_scores = {}

        for aspect, sentences in aspect_to_sentences.items():
            summaries = aspect_to_summaries.get(aspect, [])

            ranked = self.rank_sentences_for_aspect(sentences, summaries)
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
        aspect_to_polarity_sentences = entity.get("reviews", {})
        aspect_to_summaries = entity.get("summaries", {})

        topk_sentences = {}
        aspect_sentence_scores = {}

        for aspect, polarity_dict in aspect_to_polarity_sentences.items():
            summaries = aspect_to_summaries.get(aspect, [])

            topk_sentences[aspect] = {}
            aspect_sentence_scores[aspect] = {}

            # polarity_dict is like {"negative": [...], "positive": [...]}
            for polarity, sentences in polarity_dict.items():
                ranked = self.rank_sentences_for_aspect(sentences, summaries)
                topk = ranked[: self.top_k]

                topk_sentences[aspect][polarity] = [item["sentence"] for item in topk]
                aspect_sentence_scores[aspect][polarity] = ranked

        return {
            "entity_id": entity["entity_id"],
            "entity_name": entity.get("entity_name"),
            "topk_sentences": topk_sentences,
            "aspect_sentence_scores": aspect_sentence_scores,
        }
