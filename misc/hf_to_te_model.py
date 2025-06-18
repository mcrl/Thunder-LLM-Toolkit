import os
import argparse
from tqdm import tqdm
from collections import OrderedDict


import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from thunderllm.common_argument import *
from thunderllm.tokenizer import get_tokenizer
from thunderllm.model import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert arguments", allow_abbrev=False
    )

    add_model_arch_args(parser)
    add_tokenizer_args(parser)
    add_checkpoint_args(parser)

    add_device_args(parser)

    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
    )
    parser.add_argument("--save-dir", type=str, required=True)

    args = parser.parse_args()

    return args


def copy_weights(te_model_weight, hf_model_weight):
    assert (
        te_model_weight.shape == hf_model_weight.shape
    ), f"Shape mismatch: {te_model_weight.shape} != {hf_model_weight.shape}"
    # hf_model_weight.data = te_model_weight.data.clone().detach()
    # reverse
    te_model_weight.data = hf_model_weight.data.clone().detach()


def map_layer(te_layer, hf_layer):
    # qkv layernorm
    copy_weights(
        te_layer.self_attention.layernorm_qkv.layer_norm_weight,
        hf_layer.input_layernorm.weight,
    )

    # qkv weight
    copy_weights(
        te_layer.self_attention.layernorm_qkv.query_weight,
        hf_layer.self_attn.q_proj.weight,
    )
    copy_weights(
        te_layer.self_attention.layernorm_qkv.key_weight,
        hf_layer.self_attn.k_proj.weight,
    )
    copy_weights(
        te_layer.self_attention.layernorm_qkv.value_weight,
        hf_layer.self_attn.v_proj.weight,
    )

    # qkv output
    copy_weights(te_layer.self_attention.proj.weight, hf_layer.self_attn.o_proj.weight)

    # mlp layernorm
    copy_weights(
        te_layer.layernorm_mlp.layer_norm_weight,
        hf_layer.post_attention_layernorm.weight,
    )
    # mlp weight
    # gate_proj and up_proj should be concatenated to fc1

    # below is te to hf. we need to reverse its
    # size1 = te_layer.layernorm_mlp.fc1_weight.data.shape[0]
    # hf_layer.mlp.gate_proj.weight.data = (
    #     te_layer.layernorm_mlp.fc1_weight.data[: size1 // 2, :].clone().detach()
    # )
    # hf_layer.mlp.up_proj.weight.data = (
    #     te_layer.layernorm_mlp.fc1_weight.data[size1 // 2 :, :].clone().detach()
    # )

    te_layer.layernorm_mlp.fc1_weight.data = torch.cat(
        (hf_layer.mlp.gate_proj.weight.data, hf_layer.mlp.up_proj.weight.data), dim=0
    )

    copy_weights(te_layer.layernorm_mlp.fc2_weight, hf_layer.mlp.down_proj.weight)


def map_other_params(te_model, hf_model):
    copy_weights(te_model.tok_embeddings.weight, hf_model.model.embed_tokens.weight)
    copy_weights(te_model.norm.weight, hf_model.model.norm.weight)
    copy_weights(te_model.output.weight, hf_model.lm_head.weight)


args = parse_args()
src_model_path = args.checkpoint

print("Converting HF model to TE model", flush=True)
print("SRC: ", src_model_path)
print("DST: ", args.save_dir)

# load hf model
print("Loading HF model", flush=True)

hf_model = AutoModelForCausalLM.from_pretrained(src_model_path).to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(src_model_path)

# new_embed_weights = F.pad(hf_model.model.embed_tokens.weight, (0, 0, 0, 4), "constant", 0)
# new_lm_head_weights = F.pad(hf_model.lm_head.weight, (0, 0, 0, 4), "constant", 0)

# torch.nn.init.normal_(new_embed_weights[192300:, :], mean=0.0, std=0.02)
# torch.nn.init.normal_(new_lm_head_weights[192300:, :], mean=0.0, std=0.02)

# hf_model.model.embed_tokens = torch.nn.Embedding.from_pretrained(new_embed_weights, freeze=False)
# hf_model.lm_head = torch.nn.Linear(3072, 192304, bias=False)
# hf_model.lm_head.weight = torch.nn.Parameter(new_lm_head_weights)

print("Loading TE model", flush=True)
te_model = get_model(
    args.model_arch,
    args.seq_len,
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
    tokenizer.vocab_size+4,
    no_gradient_checkpointing=True,
)


print("Mapping layer weights", flush=True)
for te_layer, hf_layer in tqdm(
    zip(te_model.layers, hf_model.model.layers), total=len(te_model.layers)
):
    map_layer(te_layer, hf_layer)
print("Mapping other parameters", flush=True)
map_other_params(te_model, hf_model)
# print all parameters
for name, param in te_model.named_parameters():
    print(name, param.size(), param.dtype, param.requires_grad)
print("Save TE Model", flush=True)
te_model = te_model.bfloat16()

# save state dict
# torch.save(te_model.state_dict(), "test.pt")
param_shapes = OrderedDict(
    [(name, param.shape) for name, param in te_model.named_parameters()]
)
torch.save(
    {
        "module": te_model.state_dict(),
        "buffer_names": [],
        "optimizer": None,
        "param_shapes": param_shapes,
        "frozen_param_shapes": None,
        "shared_params": {},
        "frozen_param_fragments": None,
        "lr_scheduler": {"last_batch_iteration": 0},
        "data_sampler": None,
        "random_ltd": None,
        "sparse_tensor_module_names": set(),
        "skipped_stpes": 0,
        "global_steps": 0,
        "global_samples": 0,
        "dp_world_size": 16,
        "mp_world_size": 1,
        "ds_config": {
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 0.0002,
                    "betas": [0.9, 0.95],
                    "eps": 1e-05,
                    "weight_decay": 0.1,
                },
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 1,
                "reduce_bucket_size": 1000000000,
                "allgather_bucket_size": 1000000000,
            },
            "steps_per_print": 1000000000000,
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": 25000000.0,
                    "warmup_type": "linear",
                    "warmup_num_steps": 500.0,
                },
            },
            "train_micro_batch_size_per_gpu": 8,
            "gradient_accumulation_steps": 4,
            "bf16": {"enabled": True},
        },
        "ds_version": "0.15.1",
        "step": 0,
        "consumed_tokens": 0,
        "random_seed": 4155,
    },
    os.path.join(args.save_dir, "model.pt"),
)
tokenizer.save_pretrained(args.save_dir)
