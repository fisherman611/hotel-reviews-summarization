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
MAX_SENTENCES = config["max_sentences"]

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
        
        # Fix padding token issues (critical for Gemma and some other models)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Fallback: add a new pad token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Model-specific loading strategies
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Clear CUDA cache before loading model to prevent memory-related errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if "gemma" in self.model_name.lower():
            # Gemma: Use device_map for direct CUDA loading (avoids .to() issues)
            # Gemma prefers bfloat16 (what it was trained on) - fallback to float16 if not supported
            try:
                if self.device == "cuda":
                    # Check if bfloat16 is supported
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float16
                    
                    # Use device_map to load directly to GPU (more stable than .to())
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        device_map={"": 0},  # Load directly to GPU 0
                        low_cpu_mem_usage=True,
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                    )
                    
            except Exception as e:
                print(f"Error loading Gemma with device_map, trying fallback: {e}")
                # Fallback: Load to CPU first, then move carefully
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    low_cpu_mem_usage=True,
                )
                if self.device == "cuda":
                    self.model = self.model.half().to(self.device)
            
            self.model.eval()
            
            # Resize embeddings if we added a pad token
            if self.tokenizer.pad_token == '[PAD]':
                self.model.resize_token_embeddings(len(self.tokenizer))
                
        elif "llama" in self.model_name.lower():
            # Llama: Explicit device and dtype to avoid disk offload (better for Windows)
            if self.device == "cuda":
                dtype = torch.float16  # or torch.bfloat16 if your GPU supports it
            else:
                dtype = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=None,  # prevents implicit CPU/disk offload
                low_cpu_mem_usage=False,  # safer on Windows
            ).to(self.device)
            self.model.eval()
            
            # Resize embeddings if we added a pad token
            if self.tokenizer.pad_token == '[PAD]':
                self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            # Other models (Qwen, etc.): use automatic device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            
            # Resize embeddings if we added a pad token
            if self.tokenizer.pad_token == '[PAD]':
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Store device for easy access
        if not hasattr(self.model, 'device'):
            self.model.device = self.device

    def _clean_summary(self, summary: str) -> str:
        """Remove thinking process tags and clean up the summary."""
        # Remove <think>...</think> tags and their content
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        
        # Remove any remaining <think> or </think> tags
        summary = re.sub(r'</?think>', '', summary)
        
        # Clean up extra whitespace
        summary = ' '.join(summary.split())
        
        return summary.strip()
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache to prevent memory issues."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def print_model_info(self):
        """Print diagnostic information about the model configuration."""
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Tokenizer pad_token: {self.tokenizer.pad_token}")
        print(f"Tokenizer pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"Tokenizer eos_token: {self.tokenizer.eos_token}")
        print(f"Tokenizer eos_token_id: {self.tokenizer.eos_token_id}")
        print(f"Model vocab size: {self.model.config.vocab_size}")
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print(f"CUDA available: No")

    def _build_messages(
        self,
        entity_id: str,
        aspect: str,
        sentences: List[str],
        max_sentences: int = MAX_SENTENCES,
    ):
        if not sentences:
            return None

        selected = sentences[:max_sentences]
        bullet_list = "\n".join(f"- {s}" for s in selected)

        # Unified prompt for all models
        system_content = (
            "You are a hotel review summarizer. Create brief, accurate summaries of guest feedback. "
            "Provide ONLY the final summary without any thinking process or extra commentary."
        )
        
        example_user = (
            "Summarize hotel reviews:\n\n"
            "Hotel: Grand Plaza Hotel\n"
            "Aspect: location\n"
            "Reviews:\n"
            "- The hotel is conveniently located near the subway station.\n"
            "- Walking distance to many restaurants and shops.\n"
            "- Perfect location for exploring the city center.\n"
            "- Close to major tourist attractions."
        )
        
        example_assistant = (
            "Guests praise the convenient location with easy access to public transportation, "
            "restaurants, shops, and major tourist attractions in the city center."
        )
        
        user_prompt = (
            f"Summarize hotel reviews:\n\n"
            f"Hotel: {entity_id}\n"
            f"Aspect: {aspect}\n"
            f"Reviews:\n"
            f"{bullet_list}"
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
        max_sentences: int = MAX_SENTENCES
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
            selected = sentences[:max_sentences]
            bullet_list = "\n".join(f"- {s}" for s in selected)
            
            # Unified fallback prompt for all models
            text = (
                f"Summarize hotel reviews:\n\n"
                f"Example:\n"
                f"Hotel: Grand Plaza | Aspect: location\n"
                f"Reviews:\n"
                f"- Near subway station.\n"
                f"- Walking distance to restaurants.\n"
                f"- Great city center access.\n"
                f"Summary: Convenient location with easy access to transportation, dining, and attractions.\n\n"
                f"Hotel: {entity_id} | Aspect: {aspect}\n"
                f"Reviews:\n"
                f"{bullet_list}\n"
                f"Summary:"
            )

        model_inputs = self.tokenizer(
            [text], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Clear CUDA cache before generation (helps prevent memory-related CUDA errors)
        self._clear_cuda_cache()

        # Model-specific generation parameters
        if "gemma" in self.model_name.lower():
            # Gemma: Use greedy decoding to avoid CUDA multinomial sampling errors
            # Greedy decoding is more stable and avoids numerical instability issues
            with torch.inference_mode():
                try:
                    output_ids = self.model.generate(
                        input_ids=model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,  # Greedy decoding - most stable for Gemma
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        # Fallback: try on CPU if CUDA fails
                        print(f"CUDA error detected, attempting CPU fallback: {e}")
                        original_device = self.device
                        model_inputs_cpu = {k: v.cpu() for k, v in model_inputs.items()}
                        self.model.cpu()
                        output_ids = self.model.generate(
                            input_ids=model_inputs_cpu["input_ids"],
                            attention_mask=model_inputs_cpu["attention_mask"],
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        self.model.to(original_device)  # Move back to original device
                    else:
                        raise
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

        summary = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

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
        max_sentences: int = MAX_SENTENCES,
    ):
        if not positive_sentences and not negative_sentences:
            return None

        # Limit how many sentences we feed (half pos / half neg if both exist)
        half = max_sentences // 2 if max_sentences else None

        pos_sel = positive_sentences[:half] if half else positive_sentences
        neg_sel = negative_sentences[:half] if half else negative_sentences

        pos_block = "\n".join(f"- {s}" for s in pos_sel) if pos_sel else "(none)"
        neg_block = "\n".join(f"- {s}" for s in neg_sel) if neg_sel else "(none)"

        # Unified prompt for all models
        system_content = (
            "You are a hotel review summarizer. Write brief, balanced summaries of guest feedback. "
            "Provide ONLY the final summary without any thinking process or extra commentary."
        )
        
        example_user = (
            "Summarize hotel reviews:\n\n"
            "Hotel: Riverside Inn | Aspect: rooms\n\n"
            "Positive:\n"
            "- Spacious and well-decorated rooms.\n"
            "- Clean, comfortable beds.\n"
            "- Modern amenities.\n\n"
            "Negative:\n"
            "- Noisy air conditioning.\n"
            "- Outdated bathroom fixtures."
        )
        
        example_assistant = (
            "Guests enjoy the spacious, well-decorated rooms with comfortable beds and modern amenities. "
            "However, some noted noisy air conditioning and outdated bathroom fixtures."
        )
        
        user_prompt = (
            f"Summarize hotel reviews:\n\n"
            f"Hotel: {entity_id} | Aspect: {aspect}\n\n"
            f"Positive:\n{pos_block}\n\n"
            f"Negative:\n{neg_block}"
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
        max_sentences: int = MAX_SENTENCES,
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
            half = max_sentences // 2
            pos_sel = positive_sentences[:half] if positive_sentences else []
            neg_sel = negative_sentences[:half] if negative_sentences else []

            pos_list = "\n".join(f"- {s}" for s in pos_sel) if pos_sel else "(none)"
            neg_list = "\n".join(f"- {s}" for s in neg_sel) if neg_sel else "(none)"

            # Unified fallback prompt for all models
            text = (
                f"Summarize hotel reviews:\n\n"
                f"Example:\n"
                f"Hotel: Riverside Inn | Aspect: rooms\n"
                f"Positive:\n"
                f"- Spacious rooms.\n"
                f"- Clean beds.\n"
                f"- Modern amenities.\n"
                f"Negative:\n"
                f"- Noisy AC.\n"
                f"- Old fixtures.\n"
                f"Summary: Guests enjoy spacious rooms with modern amenities, but note noisy AC and old fixtures.\n\n"
                f"Hotel: {entity_id} | Aspect: {aspect}\n"
                f"Positive:\n{pos_list}\n"
                f"Negative:\n{neg_list}\n"
                f"Summary:"
            )

        model_inputs = self.tokenizer(
            [text], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Clear CUDA cache before generation (helps prevent memory-related CUDA errors)
        self._clear_cuda_cache()

        # Model-specific generation parameters for polarity summarization
        if "gemma" in self.model_name.lower():
            # Gemma: Use greedy decoding to avoid CUDA multinomial sampling errors
            # Greedy decoding is more stable and avoids numerical instability issues
            with torch.inference_mode():
                try:
                    output_ids = self.model.generate(
                        input_ids=model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,  # Greedy decoding - most stable for Gemma
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        # Fallback: try on CPU if CUDA fails
                        print(f"CUDA error detected, attempting CPU fallback: {e}")
                        original_device = self.device
                        model_inputs_cpu = {k: v.cpu() for k, v in model_inputs.items()}
                        self.model.cpu()
                        output_ids = self.model.generate(
                            input_ids=model_inputs_cpu["input_ids"],
                            attention_mask=model_inputs_cpu["attention_mask"],
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        self.model.to(original_device)  # Move back to original device
                    else:
                        raise
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
