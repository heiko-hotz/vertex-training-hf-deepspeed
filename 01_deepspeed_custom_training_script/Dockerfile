FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310

# Upgrade transformers and accelerate to latest version (required for Llama 3.1)
# see https://github.com/huggingface/transformers/issues/32232
# see https://github.com/hiyouga/LLaMA-Factory/issues/5400
RUN pip install --no-cache-dir -U transformers
RUN pip install --no-cache-dir -U accelerate

WORKDIR /train

COPY scripts/train.py .
COPY config/deepspeed_config.yaml .

RUN chmod +x train.py

CMD ["accelerate", "launch", "--config_file", "deepspeed_config.yaml", "train.py"]
