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
from typing import Any, List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HF_READ_TOKEN"))

with open("methods/hybrid_extractive_abstractive/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ABSTRACTIVE_MODEL = config["abstractive_model"]
MAX_NEW_TOKENS = config["max_new_tokens"]
MAX_SENTENCES = config["max_sentences"]
MAX_LENGTH = config["max_length"]
K = config["k"]  # Number of sentences to show per polarity in few-shot examples

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

        # Detailed prompt for all models
        system_content = (
            "You are an expert hotel review summarizer. Your task is to create concise and informative summaries of guest feedback.\n\n"
            "Guidelines:\n"
            "1. Synthesize the main themes and key points from the reviews\n"
            "2. Maintain objectivity and accuracy in representing guest opinions\n"
            "3. Use natural, flowing language that sounds professional\n"
            "4. Focus on the most frequently mentioned or significant points\n"
            "5. Keep the summary brief (1-2 sentences) but informative\n"
            "6. Highlight both strengths and weaknesses when present\n\n"
            "Provide ONLY the final summary without any thinking process, explanations, or extra commentary."
        )
        
        example_user = (
            "Task: Summarize the guest reviews for the following hotel aspect.\n\n"
            "Hotel: Grand Plaza Hotel\n"
            "Aspect: location\n"
            "Reviews:\n"
            "The hotel is conveniently located near the subway station.\n"
            "Walking distance to many restaurants and shops.\n"
            "Perfect location for exploring the city center.\n"
            "Close to major tourist attractions.\n\n"
            "Generate a summary:"
        )
        
        example_assistant = (
            "Guests praise the convenient location with easy access to public transportation, "
            "restaurants, shops, and major tourist attractions in the city center."
        )
        
        user_prompt = (
            f"Task: Summarize the guest reviews for the following hotel aspect.\n\n"
            f"Hotel: {entity_id}\n"
            f"Aspect: {aspect}\n"
            f"Reviews:\n"
            f"{bullet_list}\n\n"
            f"Generate a summary:"
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
            
            # Detailed fallback prompt for all models
            text = (
                f"Task: Create concise summaries of hotel guest feedback. "
                f"Synthesize main themes and key points from reviews, maintaining objectivity. "
                f"Use natural language and focus on the most significant points.\n\n"
                f"Example:\n"
                f"Hotel: Grand Plaza\n"
                f"Aspect: location\n"
                f"Reviews:\n"
                f"Near subway station.\n"
                f"Walking distance to restaurants.\n"
                f"Great city center access.\n"
                f"Summary: Convenient location with easy access to transportation, dining, and attractions.\n\n"
                f"Hotel: {entity_id}\n"
                f"Aspect: {aspect}\n"
                f"Reviews:\n"
                f"{bullet_list}\n"
                f"Summary:"
            )

        model_inputs = self.tokenizer(
            [text], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
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
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ):
        if not positive_sentences and not negative_sentences:
            return None

        # Limit how many sentences we feed (half pos / half neg if both exist)
        half = max_sentences // 2 if max_sentences else None

        pos_sel = positive_sentences[:half] if half else positive_sentences
        neg_sel = negative_sentences[:half] if half else negative_sentences

        pos_block = "\n".join(f"{s}" for s in pos_sel) if pos_sel else "(none)"
        neg_block = "\n".join(f"{s}" for s in neg_sel) if neg_sel else "(none)"

        # Detailed prompt for all models
        system_content = (
            "You are an expert hotel review summarizer. Your task is to create concise, balanced, and informative summaries of guest feedback.\n\n"
            "Guidelines:\n"
            "1. Synthesize the main points from both positive and negative feedback\n"
            "2. Maintain objectivity and balance - represent both strengths and weaknesses fairly\n"
            "3. Use natural, flowing language that sounds professional\n"
            "4. Focus on the most frequently mentioned or significant points\n"
            "5. Keep the summary brief (1-3 sentences) but informative\n"
            "6. If only positive or only negative feedback exists, summarize what's available\n"
            "7. Use connecting words (e.g., 'however', 'although', 'while') to link contrasting points\n\n"
            "Provide ONLY the final summary without any thinking process, explanations, or extra commentary."
        )
        
        # Add instruction to follow examples when few-shot examples are provided
        if few_shot_examples:
            system_content += (
                "\n\nIMPORTANT: You will be shown example summaries from similar hotels. "
                "Follow the format, style, length, and writing patterns demonstrated in these examples. "
                "Do not add any information that is not present in the provided reviews."
            )
        
        user_prompt = (
            f"Task: Summarize the guest reviews for the following hotel aspect.\n\n"
            f"Hotel: {entity_id}\n"
            f"Aspect: {aspect}\n\n"
            f"Positive Feedback:\n{pos_block}\n\n"
            f"Negative Feedback:\n{neg_block}\n\n"
            f"Generate a balanced summary:"
        )

        messages = [{"role": "system", "content": system_content}]
        
        # Add few-shot examples from retrieved similar hotels
        if few_shot_examples:
            examples_added = 0
            for example in few_shot_examples:
                # Get sentences for this aspect
                aspect_data = example.get("topk_sentences", {}).get(aspect, {})
                ex_pos_sentences = aspect_data.get("positive", [])
                ex_neg_sentences = aspect_data.get("negative", [])
                
                # Get golden summaries for this aspect
                golden_summaries = example.get("summaries", {}).get(aspect, [])
                
                if not golden_summaries:
                    continue  # Skip if no golden summary available
                
                # Format the example
                ex_pos_block = "\n".join(f"{s}" for s in ex_pos_sentences[:K]) if ex_pos_sentences else "(none)"
                ex_neg_block = "\n".join(f"{s}" for s in ex_neg_sentences[:K]) if ex_neg_sentences else "(none)"
                
                # Format all golden summaries
                if len(golden_summaries) == 1:
                    summary_text = golden_summaries[0]
                else:
                    summary_text = "\n".join(golden_summaries)
                
                example_user = (
                    f"Task: Summarize the guest reviews for the following hotel aspect.\n\n"
                    f"Hotel: {example.get('entity_name', example.get('entity_id', 'Hotel'))}\n"
                    f"Aspect: {aspect}\n\n"
                    f"Positive Feedback:\n{ex_pos_block}\n\n"
                    f"Negative Feedback:\n{ex_neg_block}\n\n"
                    f"Generate a balanced summary:"
                )
                
                messages.append({"role": "user", "content": example_user})
                messages.append({"role": "assistant", "content": summary_text})
                examples_added += 1
            
            # If no valid examples were added (all filtered out), use zero-shot
            if examples_added == 0:
                print(f"No valid examples found for {entity_id} - {aspect}. Using zero-shot.")
        # If few_shot_examples is None or empty list, use zero-shot (no examples)
        
        # Add the actual query
        messages.append({"role": "user", "content": user_prompt})
        
        return messages

    def summarize_aspect(
        self,
        entity_id: str,
        aspect: str,
        positive_sentences: List[str],
        negative_sentences: List[str],
        max_sentences: int = MAX_SENTENCES,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not positive_sentences and not negative_sentences:
            return ""

        messages = self._build_messages(
            entity_id, aspect, positive_sentences, negative_sentences,
            max_sentences=max_sentences, few_shot_examples=few_shot_examples
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

            pos_list = "\n".join(f"{s}" for s in pos_sel) if pos_sel else "(none)"
            neg_list = "\n".join(f"{s}" for s in neg_sel) if neg_sel else "(none)"

            # Build few-shot examples text
            examples_text = ""
            if few_shot_examples:
                examples_added = 0
                for i, example in enumerate(few_shot_examples):
                    aspect_data = example.get("topk_sentences", {}).get(aspect, {})
                    ex_pos = aspect_data.get("positive", [])[:K]
                    ex_neg = aspect_data.get("negative", [])[:K]
                    golden_summaries = example.get("summaries", {}).get(aspect, [])
                    
                    if not golden_summaries:
                        continue
                    
                    ex_pos_text = "\n".join(f"{s}" for s in ex_pos) if ex_pos else "(none)"
                    ex_neg_text = "\n".join(f"{s}" for s in ex_neg) if ex_neg else "(none)"
                    summary_text = golden_summaries[0] if len(golden_summaries) == 1 else "\n".join(golden_summaries)
                    
                    examples_text += (
                        f"Example {examples_added+1}:\n"
                        f"Hotel: {example.get('entity_name', example.get('entity_id', 'Hotel'))}\n"
                        f"Aspect: {aspect}\n"
                        f"Positive Feedback:\n{ex_pos_text}\n"
                        f"Negative Feedback:\n{ex_neg_text}\n"
                        f"Summary: {summary_text}\n\n"
                    )
                    examples_added += 1
                
                # If no valid examples were added, use zero-shot
                if examples_added == 0:
                    print(f"No valid examples found for {entity_id} - {aspect}. Using zero-shot.")
            # If few_shot_examples is None or empty, use zero-shot (no examples)

            if examples_text:
                # Few-shot format
                text = (
                    f"Task: Create concise, balanced summaries of hotel guest feedback. "
                    f"Synthesize main points from positive and negative reviews, maintaining objectivity. "
                    f"Use natural language and focus on the most significant points.\n\n"
                    f"IMPORTANT: Follow the format, style, length, and writing patterns demonstrated in the examples below. "
                    f"Do not add any information that is not present in the provided reviews.\n\n"
                    f"{examples_text}"
                    f"Now summarize:\n"
                    f"Hotel: {entity_id}\n"
                    f"Aspect: {aspect}\n"
                    f"Positive Feedback:\n{pos_list}\n"
                    f"Negative Feedback:\n{neg_list}\n"
                    f"Summary:"
                )
            else:
                # Zero-shot format
                text = (
                    f"Task: Create a concise, balanced summary of hotel guest feedback. "
                    f"Synthesize main points from both positive and negative reviews, maintaining objectivity. "
                    f"Use natural language and focus on the most significant points.\n\n"
                    f"Hotel: {entity_id}\n"
                    f"Aspect: {aspect}\n"
                    f"Positive Feedback:\n{pos_list}\n"
                    f"Negative Feedback:\n{neg_list}\n"
                    f"Summary:"
                )

        model_inputs = self.tokenizer(
            [text], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
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

    def summarize_entity(
        self, 
        entity: Dict[str, Any],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        entity_id = entity["entity_id"]
        golden_summaries = entity["summaries"]
        aspect_to_polarity = entity.get("topk_sentences", {})

        aspect_summaries: Dict[str, str] = {}

        for aspect, polarity_dict in aspect_to_polarity.items():
            positive_sentences = polarity_dict.get("positive", []) or []
            negative_sentences = polarity_dict.get("negative", []) or []

            summary = self.summarize_aspect(
                entity_id,
                aspect,
                positive_sentences=positive_sentences,
                negative_sentences=negative_sentences,
                few_shot_examples=few_shot_examples,
            )
            aspect_summaries[aspect] = summary

        return {
            "entity_id": entity_id,
            "reviews": entity.get("reviews"),
            "generated_summaries": aspect_summaries,
            "golden_summaries": golden_summaries,
        }

    def process(
        self, 
        entities: List[Dict[str, Any]],
        entity_examples: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process entities with optional per-entity few-shot examples.
        
        Args:
            entities: List of entities to summarize
            entity_examples: Optional dict mapping entity_id to list of retrieved examples
        
        Returns:
            List of summarized entities
        """
        results = []
        for entity in entities:
            entity_id = entity["entity_id"]
            examples = entity_examples.get(entity_id) if entity_examples else None
            result = self.summarize_entity(entity, few_shot_examples=examples)
            results.append(result)
        return results
