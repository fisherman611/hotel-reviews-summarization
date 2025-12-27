"""
Inference pipeline for finetuned aspect-based summarization models.
Loads LoRA adapters and generates summaries for test data.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.finetune_abstractive.aspect_abstractive_summarizer import (
    FinetuneAbstractiveSummarizer,
    ASPECTS,
)

# Default paths
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "space_summ_test.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetune_abstractive"


def run_inference(
    model_key: str,
    lora_path: str = None,
    base_only: bool = False,
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_name: str = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    max_entities: int = None,
):
    """
    Run inference on test data using a finetuned model.
    
    Args:
        model_key: One of "gemma3", "qwen25", "llama32"
        lora_path: Path to the LoRA adapter directory (optional if base_only=True)
        base_only: If True, use base model without LoRA adapters (zero-shot)
        data_path: Path to the test data JSON file
        output_dir: Directory to save output
        output_name: Custom name for output file
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_entities: Limit number of entities to process (for testing)
    """
    # Load test data
    print(f"Loading test data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    
    print(f"Total entities: {len(entities)}")
    
    if max_entities:
        entities = entities[:max_entities]
        print(f"Processing first {max_entities} entities")
    
    # Initialize summarizer
    print(f"\nInitializing summarizer...")
    if base_only:
        print("Running in BASE MODEL ONLY mode (zero-shot)")
    
    summarizer = FinetuneAbstractiveSummarizer(
        model_key=model_key,
        lora_path=lora_path,
        base_only=base_only,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    # Generate summaries
    print(f"\nGenerating summaries for {len(entities)} entities...")
    results = []
    
    for entity in tqdm(entities, desc="Processing entities"):
        result = summarizer.summarize_entity(entity, aspects=ASPECTS)
        results.append(result)
    
    # Prepare output
    if output_name is None:
        # Use lora path name or base model name
        if base_only:
            output_name = f"{model_key}_base_results"
        else:
            lora_name = Path(lora_path).name
            output_name = f"{lora_name}_results"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.json"
    
    # Save results as plain array
    print(f"\nSaving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with finetuned summarization models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gemma3", "qwen25", "llama32"],
        help="Model type: gemma3, qwen25, or llama32",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (optional if --base-only is set)",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Use base model without LoRA adapters (zero-shot inference)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to test data JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save output",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom name for output file",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Limit number of entities to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Validate: need lora-path if not base-only
    if not args.base_only and args.lora_path is None:
        parser.error("--lora-path is required unless --base-only is set")
    
    run_inference(
        model_key=args.model,
        lora_path=args.lora_path,
        base_only=args.base_only,
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_entities=args.max_entities,
    )


if __name__ == "__main__":
    main()

