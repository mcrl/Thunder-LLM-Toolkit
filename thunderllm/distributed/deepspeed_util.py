import deepspeed
import copy
import torch

DS_ZERO_1_BASELINE_CFG = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.00002,
            "betas": [0.9, 0.95],
            "eps": 1e-5,
            "weight_decay": 0.1,
        },
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 1000000000,
        "allgather_bucket_size": 1000000000,
    },
    "steps_per_print": 1000**4,
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_type": "linear",
        },
    },
}

DS_ZERO_2_BASELINE_CFG = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.00002,
            "betas": [0.9, 0.95],
            "eps": 1e-5,
            "weight_decay": 0.1,
        },
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True, 
        "reduce_scatter": True, 
        "reduce_bucket_size": 1000000000, 
        "allgather_bucket_size": 1000000000,
    },
    "steps_per_print": 1000**4,
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_type": "linear",
        },
    },
}

def initialize_deepspeed_model(model,
                               train_micro_batch_size_per_gpu=1,
                               gradient_accumulation_steps=1,
                               max_steps=100000,
                               lr=0.00002,
                               bf16=False,
                               warmup_lr=False):
    ds_config = copy.deepcopy(DS_ZERO_1_BASELINE_CFG)
    ds_config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps

    if ds_config["scheduler"]["type"] == "WarmupLR":
        ds_config["scheduler"]["params"]["warmup_min_lr"] = 0.0000001
        ds_config["scheduler"]["params"]["warmup_max_lr"] = lr
        ds_config["scheduler"]["params"]["warmup_num_steps"] = 1000 / \
                gradient_accumulation_steps
    elif ds_config["scheduler"]["type"] == "WarmupCosineLR":
        ds_config["scheduler"]["params"]["total_num_steps"] = (
            max_steps / gradient_accumulation_steps
        )
        ds_config['scheduler']['params']['cos_min_ratio'] = 0.1
        if warmup_lr:
            ds_config['scheduler']['params']['warmup_num_steps'] = (max_steps // 100) / gradient_accumulation_steps
        else:
            ds_config['scheduler']['params']['warmup_num_steps'] = 0
        ds_config["optimizer"]["params"]["lr"] = lr
    if bf16:
        ds_config.pop("fp16", None)
        ds_config["bf16"] = {"enabled": True}
    return deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.learnable_parameters(),
    )


def initialize_deepspeed_inference(model, **kwargs):
    world_size = kwargs.get("world_size", 1)
    tensor_parallel = kwargs.get(
        "tensor_parallel", {"tp_size": world_size, "enabled": True}
    )
    dtype = kwargs.get("dtype", torch.half)
    replace_with_kernel_inject = kwargs.get(
        "replace_with_kernel_inject", False)
    inference_config = {
        "kernel_inject": replace_with_kernel_inject,
        "dtype": dtype,
        "tensor_parallel": tensor_parallel,
    }

    ds_engine = deepspeed.init_inference(
        model,
        config=inference_config,
    )
    return ds_engine
