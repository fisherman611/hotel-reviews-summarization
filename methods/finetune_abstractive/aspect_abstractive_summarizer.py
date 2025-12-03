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

        # Load tokenizer + chat causal LM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
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
            {"role": "system", "content": "You are a helpful assistant that summarizes hotel reviews."},
            {"role": "user", "content": user_prompt}
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

        # Build chat template → raw text prompt
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Convert to model inputs
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        # Generate continuation (the summary)
        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )

        # Extract only new tokens after the prompt
        input_ids = model_inputs.input_ids
        new_tokens = [
            output[len(inp):] for inp, output in zip(input_ids, output_ids)
        ]

        summary = self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
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
