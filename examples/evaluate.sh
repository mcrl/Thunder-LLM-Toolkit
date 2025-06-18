#!/bin/bash

thunderllm_eval --model vllm \
    --model_args pretrained=meta-llama/Meta-Llama-3.1-8B,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=2 \
    --tasks thunderllm-benchmark \
    --batch_size auto \
    --output_path workspace/results \
    --gen_kwargs do_sample=False \
    -s