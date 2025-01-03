from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM
from huggingface_hub import login
import os
import subprocess
import threading
import time
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for HuggingFace models")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action="store_true",
        help="Use reentrant version of gradient checkpointing"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.0e-5,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_bnb_8bit",
        help="The optimizer to use"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 training"
    )
    
    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="The checkpoint save strategy to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Logger log level"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_text_field",
        type=str,
        default="text",
        help="The input text field name in the dataset"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--packing",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to use packing for training"
    )

    args = parser.parse_args()
    return args

def print_gpu_utilization():
    while True:
        try:
            # Run nvidia-smi command
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True
            )
            print("\nGPU Utilization:")
            for line in output.strip().split("\n"):
                idx, _, mem_used, mem_total = line.split(", ")
                print(f"GPU {idx}: {float(mem_used) / float(mem_total) * 100:.0f}% utilization, Memory: {mem_used}MB / {mem_total}MB")
        except Exception as e:
            print(f"Failed to get GPU stats: {e}")
        time.sleep(60)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Starting training script...")

    # print("Loading arguments...")
    # args = parse_args()
    # print(f"Parsed arguments: {args}")

    # config = load_config()

    if 'TRAINING_CONFIG_PATH' in os.environ:
        config_path = os.getenv('TRAINING_CONFIG_PATH')
        print(f"Loading config from {config_path}...")
        config = load_config(config_path)
    else:
        raise ValueError("TRAINING_CONFIG_PATH environment variable is not set.")

    print(f"Loaded config: {config}")
    # Start GPU monitoring in background thread
    print("Starting GPU monitoring...")
    monitor_thread = threading.Thread(target=print_gpu_utilization, daemon=True)
    monitor_thread.start()

    # Login to Hugging Face
    if "HF_TOKEN" in os.environ:
        print("Logging in to Hugging Face...")
        login(token=os.environ["HF_TOKEN"])
    else:
        print("Warning: HF_TOKEN not found in environment variables")

    
    # Load dataset
    print("Loading dataset from GCS...")
    if "DATASET_PATH" in os.environ:
        if "DATASET_NUMBER_OF_RECORDS" in os.environ:
            dataset = load_dataset(
                "json", 
                data_files=os.environ["DATASET_PATH"],
                split=f"train[:{os.environ['DATASET_NUMBER_OF_RECORDS']}]"
            )
        else:
            dataset = load_dataset(
                "json", 
                data_files=os.environ["DATASET_PATH"],
            )
        print(f"Successfully loaded dataset with {len(dataset)} examples")
    else:
        raise ValueError("DATASET_PATH environment variable is not set.")

    # config_args = {k: v for k, v in vars(args).items() if v is not None and k != 'model_name_or_path'}
    config_args = {k: v for k, v in config.items()}

    if 'MODEL_OUTPUT_DIR' in os.environ:
        output_dir = os.getenv('MODEL_OUTPUT_DIR')
        print(f"Setting output directory to {output_dir}")
        config_args.update({
            "output_dir": output_dir,
        })
    else:
        print("Warning: MODEL_OUTPUT_DIR not found in environment variables")

    print("Initializing training configuration...")
    training_args = SFTConfig(**config_args)

    # print(f"Loading model: {args.model_name_or_path}...")
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # print("Model loaded successfully")

    if 'MODEL_NAME' in os.environ:
        model_name = os.getenv('MODEL_NAME')
        print(f"Loading model: {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model loaded successfully")
    else:
        raise ValueError("MODEL_NAME environment variable is not set.")

    # print(f"Loading model: {config['model_name_or_path']}...")
    # model = AutoModelForCausalLM.from_pretrained(config['model_name_or_path'])
    # print("Model loaded successfully")

    # Initialize trainer
    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    print("Trainer initialized successfully")

    # Train
    print("Starting training process...")
    trainer.train()

    # Save the model
    print("Training completed. Saving model...")
    trainer.save_model()
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 