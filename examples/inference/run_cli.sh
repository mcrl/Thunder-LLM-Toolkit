set -a
: ${MASTER_ADDR=localhost}
: ${MASTER_PORT=4155}
: ${OMP_NUM_THREADS=16}
: ${DISTRIBUTED_BACKEND=nccl}
: ${NCCL_P2P_DISABLE=1}
set +a

CONDA_ENV=thunderllm
source ${HOME}/.bashrc
source [CONDA_PATH]/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export PATH=[CONDA_PATH]/envs/thunderllm/bin:$PATH

ckpt_path=[CKPT_PATH]
converter=${ckpt_path}/zero_to_fp32.py
tag=`cat $ckpt_path/latest`

python $converter \
    -t $tag \
    ${ckpt_path} \
    ${ckpt_path}/fp32.bin

model_path=${ckpt_path}/fp32.bin
tasks="hellaswag"
args="--model-arch hf-llama-7b \
    --checkpoint $model_path \
    --tasks $tasks \
    --seq-len 1024 \
    --tokenizer-type custom \
    --tokenizer-path [TOKENIZER_PATH] \
    --batch-size 64"

# All of belows are tested. You can choose one of them.

mpirun --np 4 thunderllm-eval $args
# thunderllm-eval $args