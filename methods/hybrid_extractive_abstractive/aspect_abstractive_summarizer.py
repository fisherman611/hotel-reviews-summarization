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

# from dotenv import load_dotenv
# load_dotenv()
# from huggingface_hub import login
# login(token=os.getenv("HF_READ_TOKEN"))

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Check if the model is Llama-based
        if self.model_name == "meta-llama/Llama-3.2-1B":
            # Llama-specific loading: explicit device and dtype to avoid disk offload
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if device == "cuda":
                dtype = torch.float16  # or torch.bfloat16 if your GPU supports it
            else:
                dtype = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,  # prevents implicit CPU/disk offload
                low_cpu_mem_usage=False,  # safer on Windows
            ).to(device)
            self.model.eval()
        else:
            # Other models (Qwen, Gemma, etc.): use automatic device mapping
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

        # Few-shot example
        example_user = (
            "Hotel: Grand Plaza Hotel\n"
            "Aspect: location\n\n"
            "Here are the customer review sentences:\n"
            "The hotel is conveniently located near the subway station.\n"
            "Walking distance to many restaurants and shops.\n"
            "Perfect location for exploring the city center.\n"
            "Close to major tourist attractions.\n\n"
            "Write a concise summary of guest opinions about the location."
        )

        example_assistant = (
            "Guests praise the hotel's convenient location, with easy access to public transportation, "
            "nearby restaurants and shops, and proximity to the city center and major tourist attractions."
        )

        user_prompt = (
            f"Hotel: {entity_id}\n"
            f"Aspect: {aspect}\n\n"
            f"Here are the customer review sentences:\n"
            f"{bullet_list}\n\n"
            f"Write a concise summary of guest opinions about the {aspect}."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes hotel reviews. "
                "Provide concise, informative summaries that capture the main themes from guest feedback.",
            },
            {"role": "user", "content": example_user},
            {"role": "assistant", "content": example_assistant},
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

        # Check if tokenizer has chat template
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for base models without chat template (few-shot format)
            selected = sentences[:30]
            bullet_list = "\n".join(f"- {s}" for s in selected)
            text = (
                f"Task: Summarize hotel reviews for a specific aspect.\n\n"
                f"Example:\n"
                f"Hotel: Grand Plaza Hotel\n"
                f"Aspect: location\n"
                f"Review sentences:\n"
                f"The hotel is conveniently located near the subway station.\n"
                f"Walking distance to many restaurants and shops.\n"
                f"Perfect location for exploring the city center.\n"
                f"Summary: Guests praise the hotel's convenient location, with easy access to public transportation, "
                f"nearby restaurants and shops, and proximity to the city center.\n\n"
                f"Now summarize:\n"
                f"Hotel: {entity_id}\n"
                f"Aspect: {aspect}\n"
                f"Review sentences:\n{bullet_list}\n"
                f"Summary:"
            )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Check if model is Llama to use appropriate generation parameters
        if self.model_name == "meta-llama/Llama-3.2-1B":
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        else:
            output_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_new_tokens
            )

        input_ids = model_inputs.input_ids
        new_tokens = [output[len(inp) :] for inp, output in zip(input_ids, output_ids)]

        summary = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[
            0
        ].strip()

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

        # Few-shot example
        example_user = (
            "Hotel: Riverside Inn\n"
            "Aspect: rooms\n\n"
            "Here are POSITIVE customer review sentences:\n"
            "The rooms are spacious and well-decorated.\n"
            "Clean and comfortable beds.\n"
            "Modern amenities and great views.\n\n"
            "Here are NEGATIVE customer review sentences:\n"
            "The air conditioning was noisy at night.\n"
            "Bathroom fixtures look outdated.\n"
            "No minibar in the room.\n\n"
            "Write a concise, balanced summary (2–3 sentences) of guest opinions "
            "about the rooms. Mention both what guests like and what they dislike, if applicable."
        )

        example_assistant = (
            "Guests appreciate the spacious, well-decorated rooms with clean, comfortable beds and modern amenities. "
            "However, some guests noted issues with noisy air conditioning at night and outdated bathroom fixtures. "
            "A few also mentioned the absence of a minibar."
        )

        user_prompt = (
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
            {"role": "user", "content": example_user},
            {"role": "assistant", "content": example_assistant},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def summarize_aspect(
        self,
        entity_id: str,
        aspect: str,
        positive_sentences: List[str],
        negative_sentences: List[str],
        max_sentences: int = 200,
    ) -> str:
        if not positive_sentences and not negative_sentences:
            return ""

        messages = self._build_messages(
            entity_id, aspect, positive_sentences, negative_sentences
        )
        if messages is None:
            return ""

        # Check if tokenizer has chat template
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for base models without chat template (few-shot format)
            half = max_sentences // 2  # max_sentences // 2
            pos_sel = positive_sentences[:half] if positive_sentences else []
            neg_sel = negative_sentences[:half] if negative_sentences else []

            pos_block = "\n".join(f"- {s}" for s in pos_sel) if pos_sel else "(none)"
            neg_block = "\n".join(f"- {s}" for s in neg_sel) if neg_sel else "(none)"

            text = (
                f"Task: Summarize hotel reviews with both positive and negative feedback.\n\n"
                f"Example:\n"
                f"Hotel: Riverside Inn\n"
                f"Aspect: rooms\n"
                f"Positive reviews:\n"
                f"The rooms are spacious and well-decorated.\n"
                f"Clean and comfortable beds.\n"
                f"Negative reviews:\n"
                f"The air conditioning was noisy at night.\n"
                f"Bathroom fixtures look outdated.\n"
                f"Summary: Guests appreciate the spacious, well-decorated rooms with clean, comfortable beds. "
                f"However, some guests noted issues with noisy air conditioning and outdated bathroom fixtures.\n\n"
                f"Now summarize:\n"
                f"Hotel: {entity_id}\n"
                f"Aspect: {aspect}\n"
                f"Positive reviews:\n{pos_block}\n\n"
                f"Negative reviews:\n{neg_block}\n\n"
                f"Summary:"
            )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Check if model is Llama to use appropriate generation parameters
        if self.model_name == "meta-llama/Llama-3.2-1B":
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        else:
            output_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_new_tokens
            )

        input_ids = model_inputs.input_ids
        new_tokens = [output[len(inp) :] for inp, output in zip(input_ids, output_ids)]

        summary = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[
            0
        ].strip()

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
            "reviews": entity.get("reviews"),
            "generated_summaries": aspect_summaries,
            "golden_summaries": golden_summaries,
        }

    # process() is inherited from AspectAbstractiveSummarizer and still works:
    # def process(self, entities): return [self.summarize_entity(e) for e in entities]
