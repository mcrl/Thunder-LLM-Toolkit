#!/bin/bash
#SBATCH --job-name=run_hellaswag
#SBATCH --output=run_hellaswag.out
#SBATCH --error=run_hellaswag.err
#SBATCH --partition=PB
#SBATCH --exclusive

set -a
: ${MASTER_ADDR=localhost}
: ${MASTER_PORT=4155}
: ${OMP_NUM_THREADS=16}
: ${DISTRIBUTED_BACKEND=nccl}
: ${NCCL_P2P_DISABLE=1}
set +a

source ${HOME}/.bashrc
# Modify here according to your conda installation path and env
source ${HOME}/miniconda3/bin/activate
conda activate thunderllm

model_path="model-path"
args="--model-arch hf-llama-1.3b --checkpoint $model_path --debug --max-gen-length 8"

# All of belows are tested. You can choose one of them.

# mpirun --np 1 thunderllm-eval $args
thunderllm-chat $args