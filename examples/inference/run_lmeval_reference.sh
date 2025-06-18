#!/bin/bash

set -a
: ${MASTER_ADDR=localhost}
: ${MASTER_PORT=4155}
: ${OMP_NUM_THREADS=16}
: ${DISTRIBUTED_BACKEND=nccl}
: ${NCCL_P2P_DISABLE=1}
set +a

source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"
conda activate <your_conda_env_name>

tasks=thunderllm-benchmark-full
# tasks=thunderllm-benchmark-pretrained # for evaluating pretrained models

hf_model=meta-llama/Llama-3.1-8B # for instance. You can change this to any model you want to evaluate.

# usage is the same as lm evaluation harness.

thunderllm-eval-harness --model hf \
    --model_args pretrained=$hf_model,dtype=auto \
    --tasks $tasks \
    --output results \
    --batch_size 1 \
    -s