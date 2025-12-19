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
from huggingface_hub import login
login(token=os.getenv("HF_READ_TOKEN"))

with open("methods/finetune_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ABSTRACTIVE_MODEL = config["abstractive_model"]
MAX_NEW_TOKENS = config["max_new_tokens"]


class AspectAbstractiveSummarizer:
    """
    Abstractive aspect summarizer using any chat-style causal LLM
    (Qwen2 Instruct, LLaMA 3 Instruct, Gemma Instruct, etc.)
    following the simple official Qwen inference pattern.
    """

    def __init__(
        self,
        model_name: str = ABSTRACTIVE_MODEL,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

    def _build_messages(
        self,
        entity_id: str,
        aspect: str,
        sentences: List[str],
        max_sentences: int = 30,
    ):
        if not sentences:
            return None

        selected = sentences[:max_sentences]
        bullet_list = "\n".join(f"- {s}" for s in selected)

        user_prompt = (
            f"You are summarizing hotel reviews.\n\n"
            f"Hotel: {entity_id}\n"
            f"Aspect: {aspect}\n\n"
            f"Here are the customer review sentences:\n"
            f"{bullet_list}\n\n"
            f"Write a concise summary of guest opinions about the {aspect}."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes hotel reviews.",
            },
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def summarize_aspect(
        self,
        entity_id: str,
        aspect: str,
        sentences: List[str],
    ) -> str:
        if not sentences:
            return ""

        messages = self._build_messages(entity_id, aspect, sentences)
        if messages is None:
            return ""

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device
        )

        output_ids = self.model.generate(
            **model_inputs, max_new_tokens=self.max_new_tokens
        )

        input_ids = model_inputs.input_ids
        new_tokens = [output[len(inp) :] for inp, output in zip(input_ids, output_ids)]

        summary = self.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        return summary

    def summarize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        entity_id = entity["entity_id"]
        aspect_to_sentences = entity.get("reviews", {})

        aspect_summaries: Dict[str, str] = {}

        for aspect, sentences in aspect_to_sentences.items():
            summary = self.summarize_aspect(entity_id, aspect, sentences)
            aspect_summaries[aspect] = summary

        return {
            "entity_id": entity_id,
            "aspect_summaries": aspect_summaries,
        }

    def process(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.summarize_entity(e) for e in entities]


class AspectPolarityAbstractiveSummarizer(AspectAbstractiveSummarizer):
    """
    Abstractive summarizer for polarity-aware data:

    Expects entity like:
    {
        "entity_id": "...",
        "entity_name": "...",  # optional
        "reviews": {
            "rooms": {
                "negative": [...],
                "positive": [...]
            },
            "location": {
                "negative": [...],
                "positive": [...]
            },
            ...
        },
        ...
    }

    Produces, per entity:
    {
        "entity_id": "...",
        "aspect_summaries": {
            "rooms": "<combined summary of pros & cons>",
            "location": "...",
            ...
        }
    }
    """

    def _build_messages(
        self,
        entity_id: str,
        aspect: str,
        positive_sentences: List[str],
        negative_sentences: List[str],
        max_sentences: int = 30,
    ):
        if not positive_sentences and not negative_sentences:
            return None

        # Limit how many sentences we feed (half pos / half neg if both exist)
        half = max_sentences // 2 if max_sentences else None

        pos_sel = positive_sentences[:half] if half else positive_sentences
        neg_sel = negative_sentences[:half] if half else negative_sentences

        pos_block = "\n".join(f"- {s}" for s in pos_sel) if pos_sel else "(none)"
        neg_block = "\n".join(f"- {s}" for s in neg_sel) if neg_sel else "(none)"

        user_prompt = (
            f"You are summarizing hotel reviews.\n\n"
            f"Hotel: {entity_id}\n"
            f"Aspect: {aspect}\n\n"
            f"Here are POSITIVE customer review sentences:\n"
            f"{pos_block}\n\n"
            f"Here are NEGATIVE customer review sentences:\n"
            f"{neg_block}\n\n"
            f"Write a concise, balanced summary (2–3 sentences) of guest opinions "
            f"about the {aspect}. Mention both what guests like and what they "
            f"dislike, if applicable."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that summarizes hotel reviews. "
                    "You always write neutral, balanced summaries that reflect both "
                    "positive and negative feedback."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def summarize_aspect(
        self,
        entity_id: str,
        aspect: str,
        positive_sentences: List[str],
        negative_sentences: List[str],
    ) -> str:
        if not positive_sentences and not negative_sentences:
            return ""

        messages = self._build_messages(
            entity_id, aspect, positive_sentences, negative_sentences
        )
        if messages is None:
            return ""

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device
        )

        output_ids = self.model.generate(
            **model_inputs, max_new_tokens=self.max_new_tokens
        )

        input_ids = model_inputs.input_ids
        new_tokens = [output[len(inp) :] for inp, output in zip(input_ids, output_ids)]

        summary = self.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

        return summary

    def summarize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        entity_id = entity["entity_id"]
        golden_summaries = entity["summaries"]
        aspect_to_polarity = entity.get("grouped_reviews", {})

        aspect_summaries: Dict[str, str] = {}

        for aspect, polarity_dict in aspect_to_polarity.items():
            positive_sentences = polarity_dict.get("positive", []) or []
            negative_sentences = polarity_dict.get("negative", []) or []

            summary = self.summarize_aspect(
                entity_id,
                aspect,
                positive_sentences=positive_sentences,
                negative_sentences=negative_sentences,
            )
            aspect_summaries[aspect] = summary

        return {
            "entity_id": entity_id,
            "generated_summaries": aspect_summaries,
            "golden_summaries": golden_summaries
        }

    # process() is inherited from AspectAbstractiveSummarizer and still works:
    # def process(self, entities): return [self.summarize_entity(e) for e in entities]
