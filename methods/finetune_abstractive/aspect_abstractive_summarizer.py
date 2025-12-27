"""
Aspect-based summarizer for finetuned models.
Uses Unsloth to load LoRA adapters and generate summaries.
"""
import os
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import json

# Import prompt formatting from prepare_data
from methods.finetune_abstractive.prepare_data import (
    SYSTEM_PROMPT,
    ASPECTS,
)

# Model configurations
MODEL_CONFIGS = {
    "gemma3": {
        "base_model": "unsloth/gemma-3-270m-it-bnb-4bit",
        "chat_template": "gemma-3",
        "load_in_4bit": True,
    },
    "qwen25": {
        "base_model": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "chat_template": "qwen-2.5",
        "load_in_4bit": True,
    },
    "llama32": {
        "base_model": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "chat_template": "llama-3.2",
        "load_in_4bit": True,
    },
}


def format_reviews_as_text(reviews: List[Dict[str, Any]]) -> str:
    """Format reviews as text paragraphs."""
    paragraphs = []
    for review in reviews:
        sentences = review.get("sentences", [])
        if sentences:
            paragraph = " ".join(sentences)
            paragraphs.append(f"- {paragraph}")
    return "\n".join(paragraphs)


def create_instruction(aspect: str, reviews_text: str) -> str:
    """Create the instruction/prompt for a specific aspect."""
    instruction = f"""Summarize the **{aspect}** aspect from the following hotel reviews.

Reviews:
{reviews_text}

Write a concise summary (15-40 words, strictly under 75 words) focusing only on {aspect}."""
    return instruction


class FinetuneAbstractiveSummarizer:
    """
    Summarizer using finetuned LoRA models loaded with Unsloth.
    Generates aspect-based summaries for hotel reviews.
    """

    def __init__(
        self,
        model_key: str,
        lora_path: str = None,
        base_only: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the summarizer with a finetuned model.
        
        Args:
            model_key: One of "gemma3", "qwen25", "llama32"
            lora_path: Path to the LoRA adapter directory (optional if base_only=True)
            base_only: If True, use base model without LoRA adapters (zero-shot)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
        """
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_key: {model_key}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        if not base_only and lora_path is None:
            raise ValueError("lora_path is required when base_only=False")
        
        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_only = base_only
        
        # Get 4-bit setting from config
        use_4bit = self.config.get("load_in_4bit", True)
        
        # Load model
        if base_only:
            # Load base model without LoRA
            print(f"Loading base model: {self.config['base_model']} (4-bit: {use_4bit})")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config['base_model'],
                max_seq_length=20000,
                load_in_4bit=use_4bit,
            )
        else:
            # Load model with LoRA adapters
            print(f"Loading finetuned model from: {lora_path} (4-bit: {use_4bit})")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=lora_path,
                max_seq_length=20000,
                load_in_4bit=use_4bit,
            )
        
        # Apply chat template
        print(f"Applying chat template: {self.config['chat_template']}")
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=self.config["chat_template"],
        )
        
        # Set to eval mode
        self.model.eval()
        print("Model loaded successfully!")

    def _create_messages(self, aspect: str, reviews_text: str) -> List[Dict[str, str]]:
        """Create chat messages for the model."""
        instruction = create_instruction(aspect, reviews_text)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        return messages

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from messages."""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only new tokens
        input_length = inputs.input_ids.shape[1]
        new_tokens = output_ids[0][input_length:]
        
        # Decode
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return response

    def summarize_aspect(
        self,
        aspect: str,
        reviews: List[Dict[str, Any]],
    ) -> str:
        """
        Generate summary for a specific aspect.
        
        Args:
            aspect: Aspect to summarize (rooms, location, etc.)
            reviews: List of review objects
            
        Returns:
            Generated summary string
        """
        if not reviews:
            return ""
        
        reviews_text = format_reviews_as_text(reviews)
        messages = self._create_messages(aspect, reviews_text)
        
        return self._generate(messages)

    def summarize_entity(
        self,
        entity: Dict[str, Any],
        aspects: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate summaries for all aspects of an entity.
        
        Args:
            entity: Entity data with reviews
            aspects: List of aspects to summarize (default: all 6)
            
        Returns:
            Dict with entity_id, reviews, generated_summaries, golden_summaries
        """
        if aspects is None:
            aspects = ASPECTS
        
        entity_id = entity.get("entity_id", "")
        reviews = entity.get("reviews", [])
        golden_summaries = entity.get("summaries", {})
        
        generated_summaries = {}
        for aspect in aspects:
            summary = self.summarize_aspect(aspect, reviews)
            generated_summaries[aspect] = summary
        
        return {
            "entity_id": entity_id,
            "reviews": reviews,
            "generated_summaries": generated_summaries,
            "golden_summaries": golden_summaries,
        }

    def process(
        self,
        entities: List[Dict[str, Any]],
        aspects: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple entities.
        
        Args:
            entities: List of entity data
            aspects: List of aspects to summarize
            
        Returns:
            List of results with generated summaries
        """
        from tqdm.auto import tqdm
        
        results = []
        for entity in tqdm(entities, desc="Generating summaries"):
            result = self.summarize_entity(entity, aspects)
            results.append(result)
        
        return results
