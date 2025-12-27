"""
Training script for aspect-based hotel review summarization using Unsloth.
Supports 4-bit QLoRA finetuning for Gemma-3, Qwen2.5, and Llama-3.2 models.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# Default paths
DATA_DIR = PROJECT_ROOT / "data" / "recipes"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetune_models"

# Model configurations
MODEL_CONFIGS = {
    "gemma3": {
        "model_name": "unsloth/gemma-3-270m-it-bnb-4bit",
        "chat_template": "gemma-3",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
    "qwen25": {
        "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "llama32": {
        "model_name": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "chat_template": "llama-3.2",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}

# Data recipe configurations
RECIPE_CONFIGS = {
    "synth_100": "synth_100.jsonl",
    "human_25": "human_25.jsonl",
    "mixed": "mixed.jsonl",
}


class TrainingHistoryCallback(TrainerCallback):
    """
    Callback to log training history (loss, learning rate, etc.) at each step.
    Saves to a JSON file for later analysis.
    """
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.history = {
            "steps": [],
            "loss": [],
            "learning_rate": [],
            "epoch": [],
            "grad_norm": [],
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.history["steps"].append(state.global_step)
            self.history["loss"].append(logs.get("loss", None))
            self.history["learning_rate"].append(logs.get("learning_rate", None))
            self.history["epoch"].append(logs.get("epoch", None))
            self.history["grad_norm"].append(logs.get("grad_norm", None))
            
            # Save after each log
            self._save_history()
    
    def _save_history(self):
        history_path = self.output_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def on_train_end(self, args, state, control, **kwargs):
        self._save_history()
        print(f"Training history saved to: {self.output_path / 'training_history.json'}")


def load_and_prepare_model(
    model_key: str,
    max_seq_length: int = 20000,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> tuple:
    """
    Load model and tokenizer with LoRA adapters.
    
    Args:
        model_key: One of "gemma3", "qwen25", "llama32"
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        
    Returns:
        Tuple of (model, tokenizer)
    """
    config = MODEL_CONFIGS[model_key]
    
    print(f"Loading model: {config['model_name']}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Apply chat template
    print(f"Applying chat template: {config['chat_template']}")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config["chat_template"],
    )
    
    return model, tokenizer


def load_and_prepare_dataset(
    recipe: str,
    tokenizer,
    data_dir: Path = DATA_DIR,
):
    """
    Load and prepare the dataset for training.
    
    Args:
        recipe: One of "synth_100", "human_25", "mixed"
        tokenizer: The tokenizer with chat template
        data_dir: Directory containing recipe JSONL files
        
    Returns:
        Prepared dataset
    """
    data_path = data_dir / RECIPE_CONFIGS[recipe]
    print(f"Loading dataset: {data_path}")
    
    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Format conversations using chat template
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    return dataset


def train(
    model_key: str,
    recipe: str,
    output_name: Optional[str] = None,
    # Training hyperparameters
    max_seq_length: int = 20000,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    lr_scheduler_type: str = "cosine",
    optim: str = "adamw_8bit",
    seed: int = 42,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    # LoRA hyperparameters
    lora_r: int = 16,
    lora_alpha: int = 32,
    # Debug options
    max_steps: int = -1,
    debug: bool = False,
):
    """
    Main training function.
    
    Args:
        model_key: One of "gemma3", "qwen25", "llama32"
        recipe: One of "synth_100", "human_25", "mixed"
        output_name: Custom output name (default: {model_key}_{recipe})
        ... (other hyperparameters)
    """
    # Validate inputs
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODEL_CONFIGS.keys())}")
    if recipe not in RECIPE_CONFIGS:
        raise ValueError(f"Unknown recipe: {recipe}. Choose from: {list(RECIPE_CONFIGS.keys())}")
    
    # Set output name
    if output_name is None:
        output_name = f"{model_key}_{recipe}"
    output_path = OUTPUT_DIR / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"Training: {output_name}")
    print(f"  Model: {model_key}")
    print(f"  Recipe: {recipe}")
    print(f"  Output: {output_path}")
    print("="*60)
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(
        model_key=model_key,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    
    # Load dataset
    dataset = load_and_prepare_dataset(
        recipe=recipe,
        tokenizer=tokenizer,
    )
    
    # Configure trainer
    config = MODEL_CONFIGS[model_key]
    
    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs if max_steps == -1 else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        optim=optim,
        weight_decay=0.01,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        output_dir=str(output_path),
        save_strategy=save_strategy if max_steps == -1 else "no",
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    # Create history callback
    history_callback = TrainingHistoryCallback(output_path)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=[history_callback],
    )
    
    # Apply train_on_responses_only
    print("Applying train_on_responses_only...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=config["instruction_part"],
        response_part=config["response_part"],
    )
    
    # Show GPU stats before training
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
    
    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()
    
    # Show final stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"\nTraining completed!")
        print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.2f} seconds")
        print(f"  Peak memory: {used_memory} GB")
    
    # Save model
    print(f"\nSaving model to: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Save training config
    training_config = {
        "model_key": model_key,
        "model_name": config["model_name"],
        "chat_template": config["chat_template"],
        "recipe": recipe,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_runtime_seconds": trainer_stats.metrics['train_runtime'],
        "train_loss": trainer_stats.metrics.get('train_loss', None),
    }
    
    with open(output_path / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    print("Done!")
    return trainer_stats


def main():
    parser = argparse.ArgumentParser(
        description="Train aspect-based summarization model with Unsloth"
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to train: gemma3, qwen25, or llama32",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        required=True,
        choices=list(RECIPE_CONFIGS.keys()),
        help="Data recipe: synth_100, human_25, or mixed",
    )
    
    # Optional arguments
    parser.add_argument("--output-name", type=str, default=None, help="Custom output name")
    parser.add_argument("--max-seq-length", type=int, default=20000, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (for debugging)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    train(
        model_key=args.model,
        recipe=args.recipe,
        output_name=args.output_name,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        max_steps=args.max_steps,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
