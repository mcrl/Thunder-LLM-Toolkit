import os
import argparse
from tqdm import tqdm

from thunderllm.common_argument import *
from thunderllm.tokenizer import get_tokenizer
from thunderllm.model import get_model

from transformers import AutoModelForCausalLM, AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="SNU-LLM Chat CLI Arguments", allow_abbrev=False
    )

    add_model_arch_args(parser)
    add_tokenizer_args(parser)
    add_checkpoint_args(parser)

    add_device_args(parser)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
    )
    parser.add_argument("--save-path", type=str, required=True)

    args = parser.parse_args()

    return args


def copy_weights(te_model_weight, hf_model_weight):
    assert te_model_weight.shape == hf_model_weight.shape
    hf_model_weight.data = te_model_weight.data.clone().detach()


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
    copy_weights(te_layer.self_attention.proj.weight,
                 hf_layer.self_attn.o_proj.weight)

    # mlp layernorm
    copy_weights(
        te_layer.layernorm_mlp.layer_norm_weight,
        hf_layer.post_attention_layernorm.weight,
    )
    # mlp weight
    # in fc1, gate_proj and up_proj are concatenated
    # in fc2, down_proj is used
    # so, we need to split the weight fc1 into two parts: gate_proj and up_proj
    size1 = te_layer.layernorm_mlp.fc1_weight.data.shape[0]
    hf_layer.mlp.gate_proj.weight.data = (
        te_layer.layernorm_mlp.fc1_weight.data[: size1 //
                                               2, :].clone().detach()
    )
    hf_layer.mlp.up_proj.weight.data = (
        te_layer.layernorm_mlp.fc1_weight.data[size1 // 2:, :].clone().detach()
    )

    copy_weights(te_layer.layernorm_mlp.fc2_weight,
                 hf_layer.mlp.down_proj.weight)


def map_other_params(te_model, hf_model):
    copy_weights(te_model.tok_embeddings.weight,
                 hf_model.model.embed_tokens.weight)
    copy_weights(te_model.norm.weight, hf_model.model.norm.weight)
    copy_weights(te_model.output.weight, hf_model.lm_head.weight)


args = parse_args()
print("Converting TE model to HF model", flush=True)
print("SRC: ", args.checkpoint)
print("DST: ", args.save_path)
tokenizer = get_tokenizer(args.tokenizer_type, args.tokenizer_path)
print("Loading TE model", flush=True)
te_model = get_model(
    args.model_arch,
    args.seq_len,
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
    tokenizer.vocab_size + 256 + 5,
    no_gradient_checkpointing=True,
)
te_model.load_checkpoint(args.checkpoint)
te_model = te_model.to("cpu")

print("Loading HF model", flush=True)
config_path = args.config_path
if config_path is None:
    print("Setting according to model architecture")
    config_path = os.path.join(
        os.path.dirname(__file__), "convert-configs", f"{args.model_arch}.json"
    )

config = AutoConfig.from_pretrained(config_path)
hf_model = AutoModelForCausalLM.from_config(config)
hf_model = hf_model.bfloat16()
hf_model = hf_model.to("cpu")

print("Mapping layer weights", flush=True)
for te_layer, hf_layer in tqdm(
    zip(te_model.layers, hf_model.model.layers), total=len(te_model.layers)
):
    map_layer(te_layer, hf_layer)
print("Mapping other parameters", flush=True)
map_other_params(te_model, hf_model)

print("Saving HF model", flush=True)
hf_model.save_pretrained(args.save_path, safe_serialization=False)
tokenizer.save_pretrained(args.save_path)
