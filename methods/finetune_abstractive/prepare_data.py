"""
Prepare data for instruction tuning following FLAN approach.
Converts hotel review data into instruction-answer pairs for finetuning.
"""
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parent_dir = "data/finetune_abstractive"
os.makedirs(parent_dir, exist_ok=True)

# Instruction templates following FLAN approach
# Use {aspect} and {reviews} placeholders for flexibility
INSTRUCTION_TEMPLATES = [
    "Summarize the {aspect}-related feedback from these hotel reviews:\n\n{reviews}",
    "Write a summary about {aspect} based on these hotel reviews:\n\n{reviews}",
    "{reviews}\n\nWhat do guests say about {aspect}?",
    "{reviews}\n\nSummarize the main points about {aspect} from the above reviews.",
    "These are hotel reviews:\n\n{reviews}\n\nWrite a brief summary focusing on {aspect}.",
    "Based on the following reviews, summarize guest opinions about {aspect}:\n\n{reviews}",
    "Please answer: What are guests saying about {aspect} in these reviews?\n\n{reviews}",
    "{reviews}\n\nQuestion: What is the overall guest feedback about {aspect}?",
    "Read these hotel reviews and summarize the {aspect}-related comments:\n\n{reviews}",
    "Hotel reviews:\n{reviews}\n\nProvide a summary of guest experiences regarding {aspect}.",
]


def format_instruction(template: str, aspect: str, reviews: str) -> str:
    """
    Format an instruction template with aspect and reviews.
    
    Args:
        template: Template string with {aspect} and {reviews} placeholders
        aspect: The aspect to summarize (e.g., "rooms", "location")
        reviews: The formatted review text
    
    Returns:
        Formatted instruction string
    """
    
    return template.replace("{aspect}", aspect).replace("{reviews}", reviews)


def chunk_reviews(reviews: List[Dict[str, Any]], chunk_size: int = 10) -> List[str]:
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
        chunk_sentences = all_sentences[i:i + chunk_size]
        bullet_list = "\n".join(f"- {s}" for s in chunk_sentences)
        chunks.append(bullet_list)
    
    return chunks if chunks else [""]


def prepare_finetuning_dataset(
    input_path: str,
    output_path: str,
    aspects: List[str] = None,
    use_all_templates: bool = True,
    template_index: int = 0,
) -> List[Dict[str, Any]]:
    """
    Prepare dataset for instruction tuning.
    
    Args:
        input_path: Path to input JSON file (e.g., test.json, train.json)
        output_path: Path to output JSONL file for finetuning
        aspects: List of aspects to include (default: all)
        use_all_templates: If True, create examples with all 10 templates
        template_index: If use_all_templates=False, which template to use (0-9)
    
    Returns:
        List of training examples
    """
    if aspects is None:
        aspects = ["rooms", "location", "service", "cleanliness", "building", "food"]
    
    # Load data
    with open(input_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    
    training_examples = []
    
    for entity in entities:
        entity_id = entity.get("entity_id", "")
        entity_name = entity.get("entity_name", "")
        reviews = entity.get("reviews", [])
        gold_summaries = entity.get("summaries", {})
        
        # Skip if no reviews
        if not reviews:
            continue
        
        # Chunk reviews instead of truncating
        review_chunks = chunk_reviews(reviews)
        
        # Create training examples for each aspect
        for aspect in aspects:
            # Get gold summaries for this aspect
            aspect_summaries = gold_summaries.get(aspect, [])
            
            # Skip if no gold summary
            if not aspect_summaries:
                continue
            
            # Use first gold summary as target
            target_summary = aspect_summaries[0]
            
            # Create examples for each chunk
            # Note: All chunks use the same target summary (the complete gold summary)
            # The model learns to generate aspect summaries from partial review data
            for chunk_idx, reviews_text in enumerate(review_chunks):
                # Format reviews with hotel name
                formatted_reviews = f"Hotel: {entity_name}\n\n{reviews_text}"
                
                if use_all_templates:
                    # Create one example per template
                    for template in INSTRUCTION_TEMPLATES:
                        instruction = format_instruction(template, aspect, formatted_reviews)
                        
                        example = {
                            "entity_id": entity_id,
                            "entity_name": entity_name,
                            "aspect": aspect,
                            "chunk_idx": chunk_idx,
                            "total_chunks": len(review_chunks),
                            "instruction": instruction,
                            "output": target_summary
                        }
                        training_examples.append(example)
                else:
                    # Use only one template
                    template = INSTRUCTION_TEMPLATES[template_index]
                    instruction = format_instruction(template, aspect, formatted_reviews)
                    
                    example = {
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "aspect": aspect,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(review_chunks),
                        "instruction": instruction,
                        "output": target_summary
                    }
                    training_examples.append(example)
    
    # Save to JSONL format (common for finetuning)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Created {len(training_examples)} training examples")
    print(f"Saved to: {output_path}")
    
    return training_examples


def prepare_inference_dataset(
    input_path: str,
    output_path: str,
    aspects: List[str] = None,
    template_index: int = 0,
) -> List[Dict[str, Any]]:
    """
    Prepare dataset for inference (no gold summaries needed).
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        aspects: List of aspects to include
        template_index: Which template to use (0-9)
    
    Returns:
        List of inference examples
    """
    if aspects is None:
        aspects = ["rooms", "location", "service", "cleanliness", "building", "food"]
    
    # Load data
    with open(input_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    
    inference_examples = []
    
    for entity in entities:
        entity_id = entity.get("entity_id", "")
        entity_name = entity.get("entity_name", "")
        reviews = entity.get("reviews", [])
        
        # Skip if no reviews
        if not reviews:
            continue
        
        # Chunk reviews instead of truncating
        review_chunks = chunk_reviews(reviews)
        
        entity_data = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "total_chunks": len(review_chunks),
            "aspects": {}
        }
        
        # Create prompts for each aspect
        for aspect in aspects:
            template = INSTRUCTION_TEMPLATES[template_index]
            
            # Create instruction for each chunk
            chunk_data = []
            for chunk_idx, reviews_text in enumerate(review_chunks):
                # Format reviews with hotel name
                formatted_reviews = f"Hotel: {entity_name}\n\n{reviews_text}"
                
                instruction = format_instruction(template, aspect, formatted_reviews)
                
                chunk_data.append({
                    "chunk_idx": chunk_idx,
                    "instruction": instruction
                })
            
            entity_data["aspects"][aspect] = chunk_data
        
        inference_examples.append(entity_data)
    
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inference_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Created inference data for {len(inference_examples)} entities")
    print(f"Saved to: {output_path}")
    
    return inference_examples


if __name__ == "__main__":
    # Example usage
    
    # Prepare training data with all templates (for diversity as in FLAN)
    print("Preparing training data with multiple templates...")
    prepare_finetuning_dataset(
        input_path="data/space_train.json",
        output_path="data/finetune_abstractive/finetuning_train_all_templates.jsonl",
        use_all_templates=True
    )
    
    # Prepare training data with single template (simpler)
    print("\nPreparing training data with single template...")
    prepare_finetuning_dataset(
        input_path="data/space_train.json",
        output_path="data/finetune_abstractive/finetuning_train_single_template.jsonl",
        use_all_templates=False,
        template_index=0
    )
    
    # Prepare inference data
    print("\nPreparing inference data...")
    prepare_inference_dataset(
        input_path="data/test.json",
        output_path="data/finetune_abstractive/inference_test.json",
        template_index=0
    )
