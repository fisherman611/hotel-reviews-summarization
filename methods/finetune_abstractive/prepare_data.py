"""
Prepare data for finetuning aspect-based hotel review summarization models.
Creates 3 data recipes and converts to chat format (JSONL) for Unsloth training.

Data Recipes:
- Recipe 1: 100 Synthetic Only
- Recipe 2: 25 Human Only  
- Recipe 3: 75 Synth + 25 Human

Chat Templates Supported:
- gemma-3, gemma3
- qwen-2.5, qwen2.5, qwen25
- llama-3.2, llama-32
"""
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RECIPES_DIR = DATA_DIR / "recipes"
TRAIN_DATA_PATH = DATA_DIR / "space_summ_train.json"  # 500 Synthetic
VAL_DATA_PATH = DATA_DIR / "space_summ_val.json"      # 25 Human

ASPECTS = ["rooms", "location", "service", "cleanliness", "building", "food"]

# Chat template markers for train_on_responses_only
CHAT_TEMPLATE_MARKERS = {
    "gemma-3": {
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
    "gemma3": {
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
    "qwen-2.5": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "qwen2.5": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "qwen25": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "llama-3.2": {
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "llama-32": {
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}

# System prompt for the summarization task
SYSTEM_PROMPT = """You are an expert abstractive summarizer. Your task is to write a summary of opinions for a specific aspect of a hotel.

**CRITICAL RULES:**
1. **NO META-LANGUAGE:** NEVER say "Guests said," "Reviewers mentioned," "The consensus is," or "Reports indicate."
   - BAD: "Guests found the location convenient."
   - GOOD: "The location is convenient."
2. **DIRECT ASSERTIONS:** State opinions as objective facts.
3. **BREVITY:** Keep it short (15-40 words), strictly under 75 words.
4. **FOCUS:** Identify the main sentiment and primary reasons for it."""


def format_reviews_as_text(reviews: List[Dict[str, Any]]) -> str:
    """
    Format reviews as text paragraphs.
    Each review's sentences are joined into a paragraph.
    """
    paragraphs = []
    for review in reviews:
        sentences = review.get("sentences", [])
        if sentences:
            paragraph = " ".join(sentences)
            paragraphs.append(f"- {paragraph}")
    return "\n".join(paragraphs)


def create_instruction(aspect: str, reviews_text: str) -> str:
    """
    Create the instruction/prompt for a specific aspect.
    """
    instruction = f"""Summarize the **{aspect}** aspect from the following hotel reviews.

Reviews:
{reviews_text}

Write a concise summary (15-40 words, strictly under 75 words) focusing only on {aspect}."""
    return instruction


def create_conversation(
    aspect: str,
    reviews: List[Dict[str, Any]],
    summary: str,
    include_system: bool = True,
) -> List[Dict[str, str]]:
    """
    Create a conversation in chat format.
    
    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    reviews_text = format_reviews_as_text(reviews)
    instruction = create_instruction(aspect, reviews_text)
    
    messages = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": summary})
    
    return messages


def load_data(path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_recipes(
    synth_data: List[Dict[str, Any]],
    human_data: List[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create 3 data recipes.
    
    Returns:
        Dict with recipe names as keys and data lists as values.
    """
    random.seed(seed)
    
    recipes = {}
    
    # Recipe 1: 100 Synthetic Only
    recipes["synth_100"] = random.sample(synth_data, min(100, len(synth_data)))
    
    # Recipe 2: 25 Human Only (all human data)
    recipes["human_25"] = human_data.copy()
    
    # Recipe 3: 75 Synth + 25 Human
    synth_sampled = random.sample(synth_data, min(75, len(synth_data)))
    recipes["mixed"] = synth_sampled + human_data.copy()
    
    return recipes


def convert_to_chat_jsonl(
    data: List[Dict[str, Any]],
    output_path: Path,
    include_system: bool = True,
) -> int:
    """
    Convert data to chat format JSONL.
    
    Each entity generates multiple training examples:
    - For each aspect × each summary variant (up to 3 per aspect)
    
    Args:
        data: List of entities with reviews and summaries
        output_path: Path to output JSONL file
        include_system: Whether to include system message
        
    Returns:
        Number of examples created
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    example_count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entity in data:
            reviews = entity.get("reviews", [])
            summaries = entity.get("summaries", {})
            entity_id = entity.get("entity_id", "")
            
            if not reviews or not summaries:
                continue
            
            for aspect in ASPECTS:
                aspect_summaries = summaries.get(aspect, [])
                if not aspect_summaries:
                    continue
                
                # Handle both list and string formats
                if isinstance(aspect_summaries, str):
                    aspect_summaries = [aspect_summaries]
                
                for summary_idx, summary in enumerate(aspect_summaries):
                    if not summary or not summary.strip():
                        continue
                    
                    conversation = create_conversation(
                        aspect=aspect,
                        reviews=reviews,
                        summary=summary.strip(),
                        include_system=include_system,
                    )
                    
                    example = {
                        "conversations": conversation,
                        "metadata": {
                            "entity_id": entity_id,
                            "aspect": aspect,
                            "summary_idx": summary_idx,
                        }
                    }
                    
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    example_count += 1
    
    return example_count


def prepare_all_recipes(
    train_path: Path = TRAIN_DATA_PATH,
    val_path: Path = VAL_DATA_PATH,
    output_dir: Path = RECIPES_DIR,
    include_system: bool = True,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare all 3 data recipes and save as JSONL.
    
    Returns:
        Dict with recipe info including paths and example counts.
    """
    print("Loading source data...")
    synth_data = load_data(train_path)
    human_data = load_data(val_path)
    
    print(f"  Synthetic samples: {len(synth_data)}")
    print(f"  Human samples: {len(human_data)}")
    
    # Create recipes
    print("\nCreating data recipes...")
    recipes = create_recipes(synth_data, human_data, seed=seed)
    
    # Convert to JSONL
    results = {}
    for recipe_name, data in recipes.items():
        output_path = output_dir / f"{recipe_name}.jsonl"
        print(f"\nProcessing recipe: {recipe_name}")
        print(f"  Entities: {len(data)}")
        
        count = convert_to_chat_jsonl(
            data=data,
            output_path=output_path,
            include_system=include_system,
        )
        
        print(f"  Training examples: {count}")
        print(f"  Output: {output_path}")
        
        results[recipe_name] = {
            "path": str(output_path),
            "entities": len(data),
            "examples": count,
        }
    
    # Save recipe metadata
    metadata_path = output_dir / "recipes_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    return results


def get_chat_template_markers(template_name: str) -> Dict[str, str]:
    """
    Get the instruction/response markers for a given chat template.
    Used with Unsloth's train_on_responses_only.
    
    Args:
        template_name: One of gemma-3, gemma3, qwen-2.5, qwen2.5, qwen25, llama-3.2, llama-32
        
    Returns:
        Dict with 'instruction_part' and 'response_part' keys
    """
    template_name = template_name.lower()
    if template_name not in CHAT_TEMPLATE_MARKERS:
        available = list(CHAT_TEMPLATE_MARKERS.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")
    return CHAT_TEMPLATE_MARKERS[template_name]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare finetuning data recipes")
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(TRAIN_DATA_PATH),
        help="Path to synthetic training data",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=str(VAL_DATA_PATH),
        help="Path to human validation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RECIPES_DIR),
        help="Output directory for recipes",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Exclude system message from conversations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    
    args = parser.parse_args()
    
    results = prepare_all_recipes(
        train_path=Path(args.train_path),
        val_path=Path(args.val_path),
        output_dir=Path(args.output_dir),
        include_system=not args.no_system,
        seed=args.seed,
    )
    
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    for recipe, info in results.items():
        print(f"  {recipe}: {info['examples']} examples from {info['entities']} entities")
