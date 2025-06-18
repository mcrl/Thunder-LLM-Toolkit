#!/bin/bash

set -a
: ${MASTER_ADDR=localhost}
: ${MASTER_PORT=41555}
: ${OMP_NUM_THREADS=1}
: ${DISTRIBUTED_BACKEND=nccl}
: ${NCCL_P2P_DISABLE=0}
set +a

# Modify here according to your conda installation path and envs
source ${HOME}/.bashrc
CONDA_ENV=thunderllm
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=1

python examples/training/train.py \
    --dataset-dir ./dataset \
    --dataset-tokenizer-type custom \
    --dataset-tokenizer-path ./models/thunder-llm  \
    --tokenizer-type custom \
    --tokenizer-path ./models/thunder-llm \
    --vocab-size 192304 \
    --model-arch te-llama3-8b \
    --global-batch-size 1024 \
    --micro-batch-size 1 \
    --seq-len 8192 \
    --lr 0.00012 \
    --max-steps 100000 \
    --allow-fp16-reduction \
    --stdout-log-interval 50 \
    --checkpoint-interval 10000 \
    --bf16 \
    --warmup-lr \
    --checkpoint-dir ./checkpoints \
    --run-name-suffix "pretrain" \
    --use-initial-weight ./models/te-LLaMA/model.pt \
    --use-wandb \
    --wandb-project-name thunderllm \
    --fp8-linear \
    --fp8-lmhead
