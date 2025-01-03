# Vertex Training with HuggingFace and DeepSpeed

This repository demonstrates how to fine-tune large language models using DeepSpeed on Google Cloud's Vertex AI platform with HuggingFace's Transformers library.

## Overview

The repository provides a complete setup for training large language models with DeepSpeed optimization on Vertex AI. It includes configuration files, training scripts, and a Jupyter notebook for orchestrating the training process.

## Repository Structure

```
01_deepspeed_custom_training_script/
├── config/
│   ├── deepspeed_config.yaml    # DeepSpeed optimization settings
│   └── training_config.yaml     # Training hyperparameters and settings
├── scripts/
│   └── train.py                 # Main training script
├── Dockerfile                   # Container definition for Vertex AI
├── vertex-notebook.ipynb        # Notebook for setting up and launching training
└── .env                        # Environment variables (for API tokens)
```

## Prerequisites

1. Google Cloud Project with Vertex AI API enabled
2. Google Cloud Storage bucket for storing training data and model artifacts
3. Hugging Face account and API token

## Setup Instructions

1. **Environment Setup**
   - Clone this repository
   - Create a `.env` file with your Hugging Face token:
     ```
     HF_TOKEN=your_token_here
     ```

2. **Configuration**
   - Adjust `training_config.yaml` for training hyperparameters
   - Modify `deepspeed_config.yaml` for DeepSpeed optimization settings
   - Update the Dockerfile if you need additional dependencies

3. **Training Data**
   - The training process expects data in JSONL format
   - Data will be automatically uploaded to your GCS bucket

## Usage

1. **Open the Jupyter Notebook**
   - Use `vertex-notebook.ipynb` to orchestrate the training process
   - The notebook guides you through:
     - Setting up GCP resources
     - Preparing training data
     - Building and pushing the training container
     - Launching the training job

2. **Configure Training**
   - Set your project-specific variables in the notebook
   - Adjust training parameters in `config/training_config.yaml`
   - Modify DeepSpeed settings in `config/deepspeed_config.yaml`

3. **Launch Training**
   - Follow the notebook cells sequentially
   - The training job will be launched on Vertex AI
   - Monitor progress through Vertex AI's interface

## Key Features

- **DeepSpeed Integration**: Optimized for efficient large model training
- **Vertex AI Compatibility**: Fully compatible with Google Cloud's ML platform
- **Configurable**: Easily adjustable training and optimization parameters
- **Containerized**: Reproducible environment using Docker

## Configuration Files

### training_config.yaml
Contains training hyperparameters including:
- Batch sizes and gradient accumulation steps
- Learning rate and scheduler settings
- Logging and evaluation parameters
- Dataset configuration

### deepspeed_config.yaml
Defines DeepSpeed-specific optimizations:
- ZeRO optimization stages
- Mixed precision settings
- Distributed training configuration
- Resource allocation

## Core Components

### Training Script (train.py)
The main training script (`scripts/train.py`) handles the entire training pipeline:

- **Configuration Management**:
  - Loads training parameters from YAML config
  - Supports environment variable overrides
  - Configurable via command-line arguments

- **Features**:
  - Integrates with HuggingFace's `SFTTrainer` for supervised fine-tuning
  - Real-time GPU utilization monitoring
  - Automatic HuggingFace Hub authentication
  - Flexible dataset loading from Google Cloud Storage
  - Supports model checkpointing and saving

- **Environment Variables**:
  - `TRAINING_CONFIG_PATH`: Path to training configuration
  - `HF_TOKEN`: HuggingFace authentication token
  - `DATASET_PATH`: Path to training data in GCS
  - `MODEL_NAME`: HuggingFace model identifier
  - `MODEL_OUTPUT_DIR`: Directory for saving model artifacts
  - `DATASET_NUMBER_OF_RECORDS` (optional): Limit number of training records

### Docker Configuration (Dockerfile)
The Dockerfile sets up the training environment:

- **Base Image**: Uses HuggingFace's PyTorch training image with CUDA 12.1 support
- **Key Components**:
  - Python 3.10
  - Latest versions of `transformers` and `accelerate` libraries
  - Pre-configured for Llama model support
  - DeepSpeed-ready environment

- **Execution**:
  - Copies training script and DeepSpeed config
  - Launches training using `accelerate` with DeepSpeed configuration
  - Automatically handles distributed training setup

## Workflow and File Organization

### 1. Initial Setup
- Install required packages:
  ```
  pip install google-cloud-aiplatform datasets py7zr pandas python-dotenv
  ```
- Set up environment variables in `.env` file with your HuggingFace token

### 2. Google Cloud Setup
The notebook orchestrates the following GCP resources:

1. **Cloud Storage Organization**:
   ```
   gs://{BUCKET_NAME}/
   ├── data/
   │   └── {DATASET_NAME}/
   │       └── train.jsonl         # Training data
   ├── {JOB_NAME}/
   │   └── {MODEL_NAME}/
   │       ├── config/            # Configuration files
   │       └── model/             # Trained model outputs
   ```

2. **Container Registry**:
   ```
   gcr.io/{PROJECT_ID}/training-containers/hf-training:latest
   ```

### 3. Training Job Flow
1. **Data Preparation**:
   - Downloads dataset from HuggingFace
   - Converts to JSONL format
   - Uploads to GCS bucket

2. **Configuration Upload**:
   - Training config is uploaded to:
     `gs://{BUCKET_NAME}/{JOB_NAME}/{MODEL_NAME}/config/training_config.yaml`

3. **Container Build**:
   - Builds Docker image with training environment
   - Pushes to Google Container Registry
   - Container includes:
     - `train.py` → `/train/train.py`
     - `deepspeed_config.yaml` → `/train/deepspeed_config.yaml`

4. **Job Execution**:
   - Launches on Vertex AI with:
     - A2 Ultra GPU (8x NVIDIA A100 80GB)
     - 250GB boot disk
     - Environment variables for configuration

### 4. Environment Variables
The training job uses these environment variables:
```
HF_TOKEN                  # HuggingFace API token
DATASET_PATH              # GCS path to training data
MODEL_OUTPUT_DIR          # GCS path for model outputs
DATASET_NUMBER_OF_RECORDS # Optional: limit training records
MODEL_NAME                # HuggingFace model identifier
TRAINING_CONFIG_PATH      # GCS path to training config
```

### 5. Monitoring and Output
- Training progress visible in Vertex AI console
- Model artifacts saved to specified GCS location
- GPU utilization metrics logged during training

## Support

For issues and questions, please open a GitHub issue in this repository.
