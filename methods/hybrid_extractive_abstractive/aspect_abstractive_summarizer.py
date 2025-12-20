import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
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

        # Model-specific loading strategies
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if "llama" in self.model_name.lower() or "gemma" in self.model_name.lower():
            # Llama and Gemma: Explicit device and dtype to avoid disk offload (better for Windows)
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

    def _clean_summary(self, summary: str) -> str:
        """Remove thinking process tags and clean up the summary."""
        # Remove <think>...</think> tags and their content
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        
        # Remove any remaining <think> or </think> tags
        summary = re.sub(r'</?think>', '', summary)
        
        # Clean up extra whitespace
        summary = ' '.join(summary.split())
        
        return summary.strip()

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

        # Model-specific prompts
        if "gemma" in self.model_name.lower():
            # Gemma prefers structured, instruction-focused prompts
            example_user = (
                "**Task**: Summarize hotel review sentences for a specific aspect.\n\n"
                "**Example Input**:\n"
                "Hotel: Grand Plaza Hotel\n"
                "Aspect: location\n"
                "Reviews:\n"
                "The hotel is conveniently located near the subway station.\n"
                "Walking distance to many restaurants and shops.\n"
                "Perfect location for exploring the city center.\n\n"
                "**Example Output**:\n"
                "Guests praise the hotel's convenient location near public transportation, restaurants, and city center."
            )
            
            example_assistant = (
                "Guests praise the hotel's convenient location near public transportation, restaurants, and city center."
            )
            
            user_prompt = (
                f"**Task**: Summarize hotel review sentences for a specific aspect.\n\n"
                f"**Input**:\n"
                f"Hotel: {entity_id}\n"
                f"Aspect: {aspect}\n"
                f"Reviews:\n"
                f"{bullet_list}\n\n"
                f"**Output**:"
            )
            
            system_content = (
                "You are a concise summarization assistant. Provide clear, factual summaries of hotel reviews. "
                "Focus on the main points without adding opinions."
            )
            
        elif "qwen" in self.model_name.lower():
            # Qwen works well with conversational, natural prompts
            example_user = (
                "Please summarize these hotel reviews about location:\n\n"
                "Hotel: Grand Plaza Hotel\n"
                "Customer feedback:\n"
                "The hotel is conveniently located near the subway station.\n"
                "Walking distance to many restaurants and shops.\n"
                "Perfect location for exploring the city center.\n"
                "Close to major tourist attractions.\n\n"
                "Provide ONLY the final summary:"
            )
            
            example_assistant = (
                "Guests appreciate the hotel's excellent location with convenient access to public transportation, "
                "dining options, shopping, and major tourist attractions in the city center."
            )
            
            user_prompt = (
                f"Please summarize these hotel reviews about {aspect}:\n\n"
                f"Hotel: {entity_id}\n"
                f"Customer feedback:\n"
                f"{bullet_list}\n\n"
                f"Provide ONLY the final summary:"
            )
            
            system_content = (
                "You are a helpful AI assistant specialized in summarizing hotel reviews. "
                "Provide ONLY the final summary. Do NOT show thinking process or use <think> tags. "
                "Give direct, concise summaries that capture guest sentiments and key points."
            )
            
        else:
            # Llama and default models - simple, direct prompts
            example_user = (
                "Summarize hotel reviews:\n\n"
                "Hotel: Grand Plaza Hotel\n"
                "Aspect: location\n"
                "Reviews:\n"
                "The hotel is conveniently located near the subway station.\n"
                "Walking distance to many restaurants and shops.\n"
                "Perfect location for exploring the city center.\n"
                "Close to major tourist attractions."
            )
            
            example_assistant = (
                "Guests praise the convenient location with easy access to transportation, restaurants, shops, and tourist attractions."
            )
            
            user_prompt = (
                f"Summarize hotel reviews:\n\n"
                f"Hotel: {entity_id}\n"
                f"Aspect: {aspect}\n"
                f"Reviews:\n"
                f"{bullet_list}"
            )
            
            system_content = (
                "You are a hotel review summarizer. Create brief, accurate summaries of guest feedback."
            )

        messages = [
            {"role": "system", "content": system_content},
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
            bullet_list = "\n".join(selected)
            
            # Model-specific fallback prompts
            if "gemma" in self.model_name.lower():
                text = (
                    f"Task: Summarize hotel reviews.\n\n"
                    f"Example:\n"
                    f"Hotel: Grand Plaza | Aspect: location\n"
                    f"Reviews: Near subway. Restaurants nearby. City center access.\n"
                    f"Summary: Convenient location near transit, dining, and city center.\n\n"
                    f"Your turn:\n"
                    f"Hotel: {entity_id} | Aspect: {aspect}\n"
                    f"Reviews: {bullet_list}\n"
                    f"Summary:"
                )
            elif "qwen" in self.model_name.lower():
                text = (
                    f"Summarize hotel reviews:\n\n"
                    f"Example - Hotel: Grand Plaza, Aspect: location\n"
                    f"Reviews: Conveniently located near subway. Walking distance to restaurants and shops. "
                    f"Perfect for exploring city center.\n"
                    f"→ Summary: Guests appreciate the excellent location with convenient access to "
                    f"public transportation, dining, and city attractions.\n\n"
                    f"Now your turn - Hotel: {entity_id}, Aspect: {aspect}\n"
                    f"Reviews: {bullet_list}\n"
                    f"→ Summary:"
                )
            else:
                # Llama and default
                text = (
                    f"Summarize reviews:\n\n"
                    f"Example:\n"
                    f"Hotel: Grand Plaza | location\n"
                    f"Near subway station. Walking distance to restaurants. Great city center access.\n"
                    f"Summary: Convenient location with easy access to transportation and attractions.\n\n"
                    f"Hotel: {entity_id} | {aspect}\n"
                    f"{bullet_list}\n"
                    f"Summary:"
                )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Model-specific generation parameters
        if "gemma" in self.model_name.lower():
            # Gemma: Lower temperature for more focused outputs
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        elif "qwen" in self.model_name.lower():
            # Qwen: Balanced parameters for natural output
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        elif "llama" in self.model_name.lower():
            # Llama: Standard sampling parameters
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
            # Default for other models
            output_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_new_tokens
            )

        input_ids = model_inputs.input_ids
        new_tokens = [output[len(inp) :] for inp, output in zip(input_ids, output_ids)]

        summary = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[
            0
        ].strip()

        # Clean up any thinking process tags (especially for Qwen)
        summary = self._clean_summary(summary)

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

        # Model-specific prompts for polarity summarization
        if "gemma" in self.model_name.lower():
            # Gemma: Structured format with clear sections
            example_user = (
                "**Task**: Create a balanced summary from positive and negative hotel reviews.\n\n"
                "**Example Input**:\n"
                "Hotel: Riverside Inn | Aspect: rooms\n\n"
                "**Positive feedback**:\n"
                "Spacious and well-decorated\n"
                "Clean, comfortable beds\n"
                "Modern amenities\n\n"
                "**Negative feedback**:\n"
                "Noisy air conditioning\n"
                "Outdated bathroom fixtures\n\n"
                "**Example Output**:\n"
                "Guests praise the spacious, well-decorated rooms with comfortable beds and modern amenities. "
                "However, some noted noisy air conditioning and outdated bathroom fixtures."
            )
            
            example_assistant = (
                "Guests praise the spacious, well-decorated rooms with comfortable beds and modern amenities. "
                "However, some noted noisy air conditioning and outdated bathroom fixtures."
            )
            
            user_prompt = (
                f"**Task**: Create a balanced summary from positive and negative hotel reviews.\n\n"
                f"**Input**:\n"
                f"Hotel: {entity_id} | Aspect: {aspect}\n\n"
                f"**Positive feedback**:\n{pos_block}\n\n"
                f"**Negative feedback**:\n{neg_block}\n\n"
                f"**Output**:"
            )
            
            system_content = (
                "You are a review summarization expert. Create balanced, objective summaries that "
                "present both positive and negative aspects clearly and concisely."
            )
            
        elif "qwen" in self.model_name.lower():
            # Qwen: Conversational, natural flow
            example_user = (
                "Please write a balanced summary for these hotel reviews:\n\n"
                "Hotel: Riverside Inn - rooms aspect\n\n"
                "What guests liked:\n"
                "The rooms are spacious and well-decorated\n"
                "Clean and comfortable beds\n"
                "Modern amenities and great views\n\n"
                "What guests disliked:\n"
                "The air conditioning was noisy at night\n"
                "Bathroom fixtures look outdated\n"
                "No minibar in the room\n\n"
                "Provide ONLY the final summary (2-3 sentences):"
            )
            
            example_assistant = (
                "Guests appreciate the spacious, well-decorated rooms featuring clean, comfortable beds and modern amenities. "
                "On the downside, several guests mentioned noisy air conditioning at night and outdated bathroom fixtures. "
                "The absence of a minibar was also noted by some guests."
            )
            
            user_prompt = (
                f"Please write a balanced summary for these hotel reviews:\n\n"
                f"Hotel: {entity_id} - {aspect} aspect\n\n"
                f"What guests liked:\n{pos_block}\n\n"
                f"What guests disliked:\n{neg_block}\n\n"
                f"Provide ONLY the final summary (2-3 sentences):"
            )
            
            system_content = (
                "You are an expert at summarizing hotel reviews. "
                "Provide ONLY the final summary. Do NOT show thinking process or use <think> tags. "
                "Write balanced, informative summaries that fairly represent both positive and negative guest experiences."
            )
            
        else:
            # Llama and default: Simple, direct format
            example_user = (
                "Summarize hotel reviews:\n\n"
                "Hotel: Riverside Inn | rooms\n\n"
                "Positive:\n"
                "Spacious, well-decorated rooms.\n"
                "Clean, comfortable beds.\n"
                "Modern amenities.\n\n"
                "Negative:\n"
                "Noisy air conditioning.\n"
                "Outdated bathroom fixtures.\n"
                "No minibar."
            )
            
            example_assistant = (
                "Guests enjoy the spacious rooms with comfortable beds and modern amenities. "
                "However, some guests complained about noisy air conditioning and outdated bathrooms."
            )
            
            user_prompt = (
                f"Summarize hotel reviews:\n\n"
                f"Hotel: {entity_id} | {aspect}\n\n"
                f"Positive:\n{pos_block}\n\n"
                f"Negative:\n{neg_block}"
            )
            
            system_content = (
                "You are a hotel review summarizer. Write brief, balanced summaries of guest feedback."
            )

        messages = [
            {"role": "system", "content": system_content},
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

            pos_list = "\n".join(pos_sel) if pos_sel else "(none)"
            neg_list = "\n".join(neg_sel) if neg_sel else "(none)"

            # Model-specific fallback prompts
            if "gemma" in self.model_name.lower():
                text = (
                    f"Task: Balanced review summary.\n\n"
                    f"Example:\n"
                    f"Hotel: Riverside Inn | rooms\n"
                    f"Pros: Spacious. Clean beds. Modern.\n"
                    f"Cons: Noisy AC. Old fixtures.\n"
                    f"Guests like the spacious, modern rooms but note noisy AC and old fixtures.\n\n"
                    f"Your turn:\n"
                    f"Hotel: {entity_id} | {aspect}\n"
                    f"Pros: {pos_list}\n"
                    f"Cons: {neg_list}\n"
                    f"Summary:"
                )
            elif "qwen" in self.model_name.lower():
                text = (
                    f"Create a balanced summary:\n\n"
                    f"Example - Riverside Inn (rooms):\n"
                    f"Positives: Spacious and well-decorated. Clean, comfortable beds. Modern amenities.\n"
                    f"Negatives: Noisy air conditioning. Outdated bathroom fixtures.\n"
                    f"Summary: Guests appreciate the spacious rooms with comfortable beds and modern amenities, "
                    f"but some noted noisy AC and outdated bathrooms.\n\n"
                    f"Now for {entity_id} ({aspect}):\n"
                    f"Positives: {pos_list}\n"
                    f"Negatives: {neg_list}\n"
                    f"Summary:"
                )
            else:
                # Llama and default
                text = (
                    f"Summarize reviews (show both pros and cons):\n\n"
                    f"Example:\n"
                    f"Hotel: Riverside Inn | rooms\n"
                    f"Good: Spacious. Clean beds. Modern amenities.\n"
                    f"Bad: Noisy AC. Old bathroom.\n"
                    f"Summary: Spacious rooms with modern amenities, but noisy AC and old bathroom.\n\n"
                    f"Hotel: {entity_id} | {aspect}\n"
                    f"Good: {pos_list}\n"
                    f"Bad: {neg_list}\n"
                    f"Summary:"
                )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Model-specific generation parameters for polarity summarization
        if "gemma" in self.model_name.lower():
            # Gemma: Lower temperature for balanced, focused outputs
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        elif "qwen" in self.model_name.lower():
            # Qwen: Natural, flowing balanced summaries
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        elif "llama" in self.model_name.lower():
            # Llama: Standard parameters for balanced output
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
            # Default for other models
            output_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_new_tokens
            )

        input_ids = model_inputs.input_ids
        new_tokens = [output[len(inp) :] for inp, output in zip(input_ids, output_ids)]

        summary = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[
            0
        ].strip()

        # Clean up any thinking process tags (especially for Qwen)
        summary = self._clean_summary(summary)

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
