import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import Any, List, Dict

# Import template system from prepare_finetuning_data
# Both files are in the same directory, so direct import works
from methods.finetune_abstractive.prepare_data import (
    INSTRUCTION_TEMPLATES,
    format_instruction,
)

with open("methods/finetune_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ABSTRACTIVE_MODEL = config["abstractive_model"]
MAX_NEW_TOKENS = config["max_new_tokens"]


class AspectAbstractiveSummarizer:
    """
    Abstractive aspect summarizer using instruction-tuned causal LLM.
    Following FLAN approach: natural language instructions for zero-shot/finetuned inference.

    This class uses a finetuned model and does NOT require aspect classification.
    Reviews are directly summarized for each aspect using instruction prompts.
    """

    def __init__(
        self,
        model_name: str = ABSTRACTIVE_MODEL,
        max_new_tokens: int = MAX_NEW_TOKENS,
        template_index: int = 0,
    ):
        """
        Args:
            model_name: Path or name of the finetuned model
            max_new_tokens: Maximum tokens to generate
            template_index: Which instruction template to use (0-9)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.template_index = template_index

        # Load tokenizer + chat causal LM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )

    def _chunk_reviews(
        self, reviews: List[Dict[str, Any]], chunk_size: int = 30
    ) -> List[str]:
        """
        Split reviews into chunks of sentences.
        Each chunk contains up to chunk_size sentences formatted as bullet list.

        Args:
            reviews: List of review dicts with sentences
            chunk_size: Number of sentences per chunk

        Returns:
            List of formatted text chunks
        """
        all_sentences = []
        for review in reviews:
            sentences = review.get("sentences", [])
            all_sentences.extend(sentences)

        # Split into chunks
        chunks = []
        for i in range(0, len(all_sentences), chunk_size):
            chunk_sentences = all_sentences[i : i + chunk_size]
            bullet_list = "\n".join(f"- {s}" for s in chunk_sentences)
            chunks.append(bullet_list)

        return chunks if chunks else [""]

    def _build_prompt(
        self,
        entity_name: str,
        aspect: str,
        reviews_text: str,
    ):
        """
        Build prompt following FLAN instruction tuning approach.

        Args:
            entity_name: Name of the hotel
            aspect: Aspect to summarize (rooms, location, service, etc.)
            reviews_text: Formatted review text (one chunk)
        """
        if not reviews_text:
            return None

        # Get instruction template and format it with aspect and reviews
        template = INSTRUCTION_TEMPLATES[self.template_index]
        formatted_reviews = f"Hotel: {entity_name}\n\n{reviews_text}"
        prompt = format_instruction(template, aspect, formatted_reviews)

        return prompt

    def _generate_from_prompt(self, prompt: str) -> str:
        """
        Generate text from a prompt string.

        Args:
            prompt: The prompt string to generate from

        Returns:
            Generated text
        """
        # Convert prompt to model inputs
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate continuation (the summary)
        output_ids = self.model.generate(
            **model_inputs, max_new_tokens=self.max_new_tokens
        )

        # Extract only new tokens after the prompt
        input_ids = model_inputs.input_ids
        new_tokens = [output[len(inp) :] for inp, output in zip(input_ids, output_ids)]

        result = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[
            0
        ].strip()

        return result

    def summarize_aspect(
        self,
        entity_name: str,
        aspect: str,
        reviews: List[Dict[str, Any]],
    ) -> str:
        """
        Generate summary for a specific aspect using instruction prompt.
        Uses chunking to handle long review lists.

        Args:
            entity_name: Name of the hotel
            aspect: Aspect to summarize
            reviews: List of review objects (original format from data)

        Returns:
            Generated summary text
        """
        if not reviews:
            return ""

        # Chunk reviews
        review_chunks = self._chunk_reviews(reviews)

        # Generate summary for each chunk
        chunk_summaries = []
        for reviews_text in review_chunks:
            prompt = self._build_prompt(entity_name, aspect, reviews_text)
            if prompt is None:
                continue

            chunk_summary = self._generate_from_prompt(prompt)
            if chunk_summary:
                chunk_summaries.append(chunk_summary)

        # If only one chunk, return directly
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # If multiple chunks, merge them using the model
        if len(chunk_summaries) > 1:
            merge_prompt = (
                f"Combine the following summaries about {aspect} for {entity_name} "
                f"into a single coherent summary. Remove redundancy and maintain key points:\n\n"
            )
            for i, summary in enumerate(chunk_summaries, 1):
                merge_prompt += f"Summary {i}: {summary}\n\n"

            final_summary = self._generate_from_prompt(merge_prompt.strip())
            return final_summary

        return ""

    def summarize_entity(
        self, entity: Dict[str, Any], aspects: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate summaries for all aspects of an entity.

        Args:
            entity: Entity data with reviews (original format)
            aspects: List of aspects to summarize (default: all)

        Returns:
            Dictionary with entity_id and aspect_summaries
        """
        if aspects is None:
            aspects = [
                "rooms",
                "location",
                "service",
                "cleanliness",
                "building",
                "food",
            ]

        entity_id = entity.get("entity_id", "")
        entity_name = entity.get("entity_name", "")
        reviews = entity.get("reviews", [])

        aspect_summaries: Dict[str, str] = {}

        for aspect in aspects:
            summary = self.summarize_aspect(entity_name, aspect, reviews)
            aspect_summaries[aspect] = summary

        return {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "aspect_summaries": aspect_summaries,
        }

    def process(
        self, entities: List[Dict[str, Any]], aspects: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple entities.

        Args:
            entities: List of entity data
            aspects: List of aspects to summarize

        Returns:
            List of results with summaries
        """
        return [self.summarize_entity(e, aspects) for e in entities]
