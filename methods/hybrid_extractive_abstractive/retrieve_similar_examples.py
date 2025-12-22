import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import torch
from tqdm.auto import tqdm

with open("methods/hybrid_extractive_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

RETRIEVAL_MODEL = config["retrieval_model"]
RERANKER_MODEL = config["reranker_model"]
K = config["k"]
NUM_EXAMPLES = config["num_examples"]
# Paths
RETRIEVE_DATA_PATH = Path(config["retrieve_data_path"])
INDEX_DIR = Path(config["index_dir"])
os.makedirs(INDEX_DIR, exist_ok=True)

@dataclass
class RetrievedExample:
    """A retrieved example with its scores."""
    entity_id: str
    entity_name: str
    topk_sentences: Dict[str, Dict[str, List[str]]]  # {aspect: {polarity: sentences}}
    summaries: Dict[str, List[str]]  # Golden summaries
    bm25_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 (sparse) and dense embeddings.
    Uses FAISS for efficient dense vector search and a cross-encoder for reranking.
    """
    
    def __init__(
        self,
        dense_model_name: str = RETRIEVAL_MODEL,
        reranker_model_name: str = RERANKER_MODEL,
        device: str = None,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
    ):
        self.dense_model_name = dense_model_name
        self.reranker_model_name = reranker_model_name
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load models
        print(f"Loading dense encoder: {dense_model_name}")
        self.dense_encoder = SentenceTransformer(dense_model_name, device=self.device)
        
        print(f"Loading reranker: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)
        
        # These will be set when building or loading the index
        self.retrieve_data: List[Dict[str, Any]] = []
        self.entity_texts: List[str] = []  # Concatenated review text per entity
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        
    def _get_entity_text(self, entity: Dict[str, Any]) -> str:
        """
        Convert entity's topk_sentences into a single text representation.
        This text is used for both BM25 and dense retrieval.
        """
        texts = []
        topk_sentences = entity.get("topk_sentences", {})
        
        for aspect, polarity_dict in topk_sentences.items():
            for polarity, sentences in polarity_dict.items():
                if sentences:
                    # Add aspect and polarity context
                    texts.append(f"{aspect} {polarity}: " + " ".join(sentences))
        
        return " ".join(texts)
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def build_index(
        self,
        retrieve_data: List[Dict[str, Any]],
        index_dir: Path = INDEX_DIR,
        save_index: bool = True,
    ):
        """
        Build BM25 and FAISS indices from retrieve data.
        
        Args:
            retrieve_data: List of entities from prepare_retrieve_data.py output
            index_dir: Directory to save the index files
            save_index: Whether to save the index to disk
        """
        self.retrieve_data = retrieve_data
        
        print("Building entity text representations...")
        self.entity_texts = [self._get_entity_text(entity) for entity in tqdm(retrieve_data)]
        
        # Build BM25 index
        print("Building BM25 index...")
        tokenized_texts = [self._tokenize_for_bm25(text) for text in self.entity_texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Build dense embeddings and FAISS index
        print("Encoding entities for dense retrieval...")
        entity_embeddings = self.dense_encoder.encode(
            self.entity_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        # Create FAISS index (using Inner Product for normalized embeddings = cosine similarity)
        embedding_dim = entity_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(entity_embeddings.astype(np.float32))
        
        print(f"Index built with {len(retrieve_data)} entities, embedding dim: {embedding_dim}")
        
        if save_index:
            self.save_index(index_dir)
    
    def save_index(self, index_dir: Path = INDEX_DIR):
        """Save the index to disk."""
        index_dir = Path(index_dir)
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(index_dir / "faiss_index.bin"))
        
        # Save BM25 and metadata
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        
        with open(index_dir / "entity_texts.pkl", "wb") as f:
            pickle.dump(self.entity_texts, f)
        
        with open(index_dir / "retrieve_data.json", "w", encoding="utf-8") as f:
            json.dump(self.retrieve_data, f, ensure_ascii=False, indent=2)
        
        print(f"Index saved to {index_dir}")
    
    def load_index(self, index_dir: Path = INDEX_DIR):
        """Load the index from disk."""
        index_dir = Path(index_dir)
        
        print(f"Loading index from {index_dir}...")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_dir / "faiss_index.bin"))
        
        # Load BM25 and metadata
        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        
        with open(index_dir / "entity_texts.pkl", "rb") as f:
            self.entity_texts = pickle.load(f)
        
        with open(index_dir / "retrieve_data.json", "r", encoding="utf-8") as f:
            self.retrieve_data = json.load(f)
        
        print(f"Index loaded: {len(self.retrieve_data)} entities")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization of scores to [0, 1]."""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def retrieve(
        self,
        query_entity: Dict[str, Any],
        top_k: int = NUM_EXAMPLES,
        top_k_candidates: int = 50,
        rerank: bool = True,
        exclude_entity_ids: Optional[List[str]] = None,
    ) -> List[RetrievedExample]:
        """
        Retrieve similar examples for a query entity using hybrid search.
        
        Args:
            query_entity: The test entity to find similar examples for.
                         Should have "reviews" field with list of review dicts.
            top_k: Number of final examples to return after reranking
            top_k_candidates: Number of candidates to retrieve before reranking
            rerank: Whether to use cross-encoder reranking
            exclude_entity_ids: Entity IDs to exclude from results (e.g., the query itself)
        
        Returns:
            List of RetrievedExample objects sorted by relevance
        """
        if self.bm25 is None or self.faiss_index is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        exclude_entity_ids = exclude_entity_ids or []
        
        # Build query text from the entity's reviews
        query_text = self._build_query_text(query_entity)
        
        # --- BM25 Search ---
        query_tokens = self._tokenize_for_bm25(query_text)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        
        # --- Dense Search with FAISS ---
        query_embedding = self.dense_encoder.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        
        # Get all dense scores (for combining with BM25)
        dense_scores, _ = self.faiss_index.search(query_embedding, len(self.retrieve_data))
        dense_scores = dense_scores[0]  # Shape: (n_entities,)
        
        # --- Combine Scores (Hybrid) ---
        # Normalize both score arrays
        bm25_normalized = self._normalize_scores(bm25_scores)
        dense_normalized = self._normalize_scores(dense_scores)
        
        # Combine with weights
        hybrid_scores = (self.bm25_weight * bm25_normalized + 
                        self.dense_weight * dense_normalized)
        
        # Get top candidates
        candidate_indices = np.argsort(hybrid_scores)[::-1]
        
        # Filter out excluded entities and get top_k_candidates
        candidates = []
        for idx in candidate_indices:
            entity = self.retrieve_data[idx]
            if entity["entity_id"] not in exclude_entity_ids:
                candidates.append(RetrievedExample(
                    entity_id=entity["entity_id"],
                    entity_name=entity.get("entity_name", ""),
                    topk_sentences=entity.get("topk_sentences", {}),
                    summaries=entity.get("summaries", {}),
                    bm25_score=float(bm25_scores[idx]),
                    dense_score=float(dense_scores[idx]),
                    hybrid_score=float(hybrid_scores[idx]),
                ))
            if len(candidates) >= top_k_candidates:
                break
        
        # --- Reranking with Cross-Encoder ---
        if rerank and len(candidates) > 0:
            print(f"Reranking {len(candidates)} candidates...")
            
            # Prepare pairs for reranking
            pairs = [(query_text, self._get_entity_text_for_rerank(c)) for c in candidates]
            
            # Get rerank scores
            rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
            
            # Update candidates with rerank scores
            for i, score in enumerate(rerank_scores):
                candidates[i].rerank_score = float(score)
            
            # Sort by rerank score
            candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return candidates[:top_k]
    
    def _build_query_text(self, query_entity: Dict[str, Any]) -> str:
        """
        Build query text from an entity's reviews.
        For test entities, we use the raw reviews (joined sentences).
        """
        reviews = query_entity.get("reviews", [])
        
        if not reviews:
            # Fallback: maybe it already has topk_sentences
            return self._get_entity_text(query_entity)
        
        # Join all sentences from all reviews
        all_sentences = []
        for review in reviews:
            sentences = review.get("sentences", [])
            all_sentences.extend(sentences)
        
        return " ".join(all_sentences)
    
    def _get_entity_text_for_rerank(self, example: RetrievedExample) -> str:
        """Get text representation for reranking."""
        texts = []
        for aspect, polarity_dict in example.topk_sentences.items():
            for polarity, sentences in polarity_dict.items():
                if sentences:
                    texts.append(f"{aspect} {polarity}: " + " ".join(sentences))
        return " ".join(texts)
    
    def retrieve_for_aspect(
        self,
        query_entity: Dict[str, Any],
        aspect: str,
        polarity: str = None,
        top_k: int = NUM_EXAMPLES,
    ) -> List[RetrievedExample]:
        """
        Retrieve examples specifically relevant to a given aspect (and optionally polarity).
        Useful for aspect-specific few-shot prompting.
        """
        # Get general retrieved examples first
        examples = self.retrieve(query_entity, top_k=top_k * 2)
        
        # Filter/prioritize those that have good coverage for the target aspect
        scored_examples = []
        for ex in examples:
            aspect_sentences = ex.topk_sentences.get(aspect, {})
            
            if polarity:
                sentences = aspect_sentences.get(polarity, [])
                coverage_score = len(sentences)
            else:
                coverage_score = sum(len(s) for s in aspect_sentences.values())
            
            # Only include if it has relevant content
            if coverage_score > 0:
                scored_examples.append((ex, coverage_score))
        
        # Sort by coverage then by rerank score
        scored_examples.sort(key=lambda x: (x[1], x[0].rerank_score), reverse=True)
        
        return [ex for ex, _ in scored_examples[:top_k]]


def build_retrieval_index(
    retrieve_data_path: Path = RETRIEVE_DATA_PATH,
    index_dir: Path = INDEX_DIR,
):
    """Build and save the retrieval index from prepared retrieve data."""
    print(f"Loading retrieve data from {retrieve_data_path}...")
    with open(retrieve_data_path, "r", encoding="utf-8") as f:
        retrieve_data = json.load(f)
    
    retriever = HybridRetriever()
    retriever.build_index(retrieve_data, index_dir=index_dir, save_index=True)
    
    return retriever


def load_retriever(index_dir: Path = INDEX_DIR) -> HybridRetriever:
    """Load a pre-built retriever from disk."""
    retriever = HybridRetriever()
    retriever.load_index(index_dir)
    return retriever


def format_few_shot_examples(
    examples: List[RetrievedExample],
    aspect: str,
    max_examples: int = 3,
) -> str:
    """
    Format retrieved examples into a few-shot prompt string.
    
    Args:
        examples: Retrieved examples
        aspect: Target aspect to show examples for
        max_examples: Maximum number of examples to include
    
    Returns:
        Formatted few-shot examples string
    """
    few_shot_parts = []
    
    for i, example in enumerate(examples[:max_examples]):
        # Get sentences for this aspect
        aspect_data = example.topk_sentences.get(aspect, {})
        pos_sentences = aspect_data.get("positive", [])
        neg_sentences = aspect_data.get("negative", [])
        
        # Get golden summaries for this aspect
        golden_summaries = example.summaries.get(aspect, [])
        
        if not golden_summaries:
            continue  # Skip if no golden summaries available
        
        # Format the example
        pos_block = "\n".join(f"- {s}" for s in pos_sentences[:K]) if pos_sentences else "(none)"
        neg_block = "\n".join(f"- {s}" for s in neg_sentences[:K]) if neg_sentences else "(none)"
        
        # Format all golden summaries
        if len(golden_summaries) == 1:
            summary_block = f"Golden Summary: {golden_summaries[0]}"
        else:
            summary_block = "Golden Summaries:\n" + "\n".join(f"- {s}" for s in golden_summaries)
        
        example_text = (
            f"Example {i+1}:\n"
            f"Hotel: {example.entity_name or example.entity_id} | Aspect: {aspect}\n\n"
            f"Positive:\n{pos_block}\n\n"
            f"Negative:\n{neg_block}\n\n"
            f"{summary_block}"
        )
        few_shot_parts.append(example_text)
    
    return "\n\n---\n\n".join(few_shot_parts)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build or test retrieval index")
    parser.add_argument("--build", action="store_true", help="Build the index")
    parser.add_argument("--test", action="store_true", help="Test retrieval")
    parser.add_argument("--retrieve_data", type=str, default=str(RETRIEVE_DATA_PATH),
                       help="Path to retrieve data JSON")
    parser.add_argument("--index_dir", type=str, default=str(INDEX_DIR),
                       help="Directory to save/load index")
    parser.add_argument("--test_data", type=str, default="data/test.json",
                       help="Path to test data for testing retrieval")
    
    args = parser.parse_args()
    
    if args.build:
        retriever = build_retrieval_index(
            retrieve_data_path=Path(args.retrieve_data),
            index_dir=Path(args.index_dir),
        )
        print("Index built successfully!")
    
    if args.test:
        # Load retriever
        retriever = load_retriever(Path(args.index_dir))
        
        # Load test data
        with open(args.test_data, "r", encoding="utf-8") as f:
            test_entities = json.load(f)
        
        # Test on first entity
        if test_entities:
            test_entity = test_entities[0]
            print(f"\nTesting retrieval for entity: {test_entity.get('entity_name', test_entity['entity_id'])}")
            
            results = retriever.retrieve(
                test_entity,
                top_k=3,
                exclude_entity_ids=[test_entity["entity_id"]],
            )
            
            print(f"\nTop {len(results)} retrieved examples:")
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result.entity_name or result.entity_id}")
                print(f"   BM25 score: {result.bm25_score:.4f}")
                print(f"   Dense score: {result.dense_score:.4f}")
                print(f"   Hybrid score: {result.hybrid_score:.4f}")
                print(f"   Rerank score: {result.rerank_score:.4f}")
                
                # Show aspects covered
                aspects = list(result.topk_sentences.keys())
                print(f"   Aspects: {', '.join(aspects)}")
            
            # Test few-shot formatting
            print("\n" + "="*50)
            print("Few-shot examples for 'rooms' aspect:")
            print("="*50)
            few_shot = format_few_shot_examples(results, aspect="rooms", max_examples=2)
            print(few_shot)
