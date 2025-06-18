import thunderllm
import arguments
import torch
import torch.nn.functional as F
import time
import math
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(model, optimizer_state_dict):
    for name, p in model.named_parameters():
        print(f"{name}, {p.shape}, {p.numel() / 1e6:.2f}M, {p.requires_grad}, {p.dtype}")
    print(optimizer_state_dict.keys())

def is_pow2(x: int) -> bool:
    return x != 0 and (x & (x - 1)) == 0

def generate_inter_document_mask(doc_ids, num_heads):
    mbs, seq_len = doc_ids.size()

    doc_pair_ids = torch.where(doc_ids > 0, (doc_ids + 1) // 2, 0)
    doc_match = doc_pair_ids.unsqueeze(1) == doc_pair_ids.unsqueeze(2)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).expand(mbs, -1, -1)

    non_pad = doc_ids != 0
    non_pad_mask = non_pad.unsqueeze(1) & non_pad.unsqueeze(2)

    final_mask = doc_match & causal_mask & non_pad_mask

    neginf = -(2**50)
    final_mask = torch.where(final_mask == 1, 0, neginf).to(dtype=torch.bfloat16, device='cuda')
    final_mask = final_mask.unsqueeze(1).expand(mbs, num_heads, seq_len, seq_len)
    final_mask.requires_grad = False

    return final_mask

def sft_loss(doc_ids, logits, targets):
    mbs, seq_len = doc_ids.size()

    answer_ids = torch.where((doc_ids % 2 == 0) & (doc_ids > 0), doc_ids, 0)
    question_ids = torch.where(doc_ids % 2 == 1, doc_ids, 0)

    last_question_tokens = torch.zeros_like(question_ids)

    for i in range(mbs):
        current_question_ids = question_ids[i]
        current_answer_ids = answer_ids[i]

        unique_qids = torch.unique(current_question_ids)

        for qid in unique_qids:
            if qid == 0:
                continue

            aid = qid + 1
            if not aid in current_answer_ids:
                positions = (current_question_ids == qid).nonzero()
                last_position = positions[-1]
                last_question_tokens[i, last_position] = 1
    
    mask = (answer_ids > 0) | last_question_tokens

    logits = logits.view(mbs * seq_len, -1)
    targets = targets.view(-1)
    mask = mask.view(-1)

    logits = logits[mask]
    targets = targets[mask]

    return thunderllm.util.CEloss(logits, targets)

def dpo_loss(doc_ids, logits, targets, beta=0.1, gamma=0.5):
    mbs, seq_len = doc_ids.size()
    assert mbs == 1, "DPO is only supported for batch size 1"

    doc_ids = doc_ids[0]

    chosen_question_ids = torch.where(doc_ids % 4 == 1, doc_ids, 0)
    chosen_ids = torch.where((doc_ids % 4 == 2), doc_ids, 0)
    rejected_question_ids = torch.where(doc_ids % 4 == 3, doc_ids, 0)
    rejected_ids = torch.where((doc_ids % 4 == 0) & (doc_ids > 0), doc_ids, 0)
    
    last_question_tokens = torch.zeros_like(chosen_question_ids)
    for qid in torch.unique(chosen_question_ids):
        if qid == 0:
            continue

        aid = qid + 1
        if not aid in chosen_ids:
            positions = (chosen_ids == qid).nonzero()
            last_position = positions[-1]
            last_question_tokens[last_position] = 1

    chosen_mask = (chosen_ids > 0) | (last_question_tokens == 1)

    last_question_tokens = torch.zeros_like(rejected_question_ids)
    for qid in torch.unique(rejected_question_ids):
        if qid == 0:
            continue

        aid = qid + 1
        if not aid in rejected_ids:
            positions = (rejected_ids == qid).nonzero()
            last_position = positions[-1]
            last_question_tokens[last_position] = 1

    rejected_mask = (rejected_ids > 0) | (last_question_tokens == 1)

    chosen_logits = logits[:, chosen_mask, :]
    rejected_logits = logits[:, rejected_mask, :]

    chosen_ids = targets[:, chosen_mask]
    rejected_ids = targets[:, rejected_mask]

    chosen_logps = torch.gather(chosen_logits.log_softmax(-1), dim=2, index=chosen_ids.unsqueeze(2)).squeeze(2).view(-1)
    rejected_logps = torch.gather(rejected_logits.log_softmax(-1), dim=2, index=rejected_ids.unsqueeze(2)).squeeze(2).view(-1)

    num_bins = torch.max(doc_ids) + 1

    accum_chosen_logps = torch.zeros(
        num_bins,
        dtype=chosen_logps.dtype,
        device=chosen_logps.device
    )
    accum_chosen_logps.index_add_(0, doc_ids[chosen_mask], chosen_logps)

    accum_rejected_logps = torch.zeros(
        num_bins,
        dtype=rejected_logps.dtype,
        device=rejected_logps.device
    )
    accum_rejected_logps.index_add_(0, doc_ids[rejected_mask], rejected_logps)

    accum_chosen_logps = accum_chosen_logps[torch.unique(doc_ids[chosen_mask])]
    accum_rejected_logps = accum_rejected_logps[torch.unique(doc_ids[rejected_mask])]

    pi_logratios = accum_chosen_logps - accum_rejected_logps
    gamma_logratios = gamma / beta
    logits = pi_logratios - gamma_logratios
    loss = (-F.logsigmoid(beta * logits)).mean()

    return loss

if __name__ == "__main__":
    args = arguments.parse_args()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = args.allow_fp16_reduction

    process_grid = thunderllm.distributed.initialize_distributed_from_envvar()

    dataset_tokenizer = thunderllm.tokenizer.get_tokenizer(
        tokenizer_type=args.dataset_tokenizer_type,
        tokenizer_path=args.dataset_tokenizer_path)

    if args.use_pretrained_hf_model != "":
        model, tokenizer, config = thunderllm.model.get_hf_model_from_pretrained(
            checkpoint_path=args.use_pretrained_hf_model,
        )
    else:
        tokenizer = thunderllm.tokenizer.get_tokenizer(
            tokenizer_type=args.tokenizer_type,
            tokenizer_path=args.tokenizer_path)
        model = thunderllm.model.get_model(model_arch=args.model_arch,
                                    max_seq_len=args.seq_len,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    vocab_size=tokenizer.vocab_size if args.vocab_size is None else args.vocab_size,
                                    fp8_linear=args.fp8_linear,
                                    fp8_mha=args.fp8_mha,
                                    fp8_lmhead=args.fp8_lmhead)

        if args.use_initial_weight != '':
            model.load_checkpoint(args.use_initial_weight)

    thunderllm.util.logger.initialize_logger(
        use_wandb=args.use_wandb,
        run_name=f"{args.model_arch}:{args.run_name_suffix}",
        wandb_project_name=args.wandb_project_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        wandb_log_interval=args.wandb_log_interval,
        stdout_log_interval=args.stdout_log_interval,
        profile=args.profile,
        profile_dir=args.profile_dir,
        random_seed=args.seed,
        model=model
    )

    model_engine, optimizer, _, lr_scheduler = thunderllm.distributed.initialize_deepspeed_model(
        model,
        train_micro_batch_size_per_gpu=args.micro_batch_size,
        gradient_accumulation_steps=args.global_batch_size // args.micro_batch_size // process_grid.world_size,
        max_steps=args.max_steps,
        lr=args.lr,
        bf16=args.bf16,
        warmup_lr=args.warmup_lr)

    if process_grid.rank == 0:
        model = model_engine.module
        optimizer_state_dict = model_engine.optimizer.state_dict()
        print(f"# Parameters: {count_parameters(model) / 1e9:.2f} B")
        print_parameters(model, optimizer_state_dict)

    n_comsumed_file_global = 0
    start_step = 1
    if args.resume != '':
        state, checkpoint_num_ranks = thunderllm.util.checkpoint.load_checkpoint(
            model_engine=model_engine,
            checkpoint_path=args.resume,
        )

        dataset_meta = thunderllm.dataset._get_meta_info(args.dataset_dir)
        tokens_per_file = dataset_meta["samples_per_file"] * dataset_meta["block_size"]
        n_consumed_file = math.ceil(state["consumed_tokens"] / tokens_per_file)
        n_comsumed_file_global = n_consumed_file * checkpoint_num_ranks
        n_consumed_file_local = math.ceil(n_comsumed_file_global / process_grid.world_size)

        start_step = state["step"] + 1
        args.seed = state["random_seed"]

        print(f"Resuming from step {start_step}")

        thunderllm.logger.set_consumed_tokens(n_consumed_file_local * tokens_per_file)

    dataset = thunderllm.dataset.get_dataset(dataset_path=args.dataset_dir,
                                        format='.parquet',
                                        streaming=True,
                                        rank=process_grid.rank,
                                        world_size=process_grid.world_size,
                                        seed=args.seed,
                                        num_skip_files=n_comsumed_file_global,
                                        drop_last=True)

    train_dataloader = thunderllm.dataloader.get_dataloader(
        dataloader_type=args.dataloader_type, dataset=dataset, batch_size=args.micro_batch_size)
    train_data_iterator = iter(train_dataloader)

    te_fp8 = ('te-' in args.model_arch) and (args.fp8_linear or args.fp8_mha or args.fp8_dpa)
    backend = "te" if ('te-' in args.model_arch) else "hf"

    if backend == "te":
        num_attention_heads = model.params.n_heads
    else:
        num_attention_heads = config.num_attention_heads

    if te_fp8:
        print(f"Using TE FP8: {args.fp8_linear=}, {args.fp8_dpa=}, {args.fp8_mha=}")
        fp8_recipe = recipe.DelayedScaling(
            margin=0, fp8_format=recipe.Format.HYBRID,
            amax_history_len=args.fp8_scale_delay,
            fp8_dpa=args.fp8_dpa, fp8_mha=args.fp8_mha)

    for step in range(start_step, args.max_steps + 1):
        data = next(train_data_iterator)
        data = {k: v.to('cuda') for k, v in data.items()}

        if args.train_mode == 'sft' or args.train_mode == 'dpo':
            input_ids = data["input_ids"][:, :-1]
            document_ids = data["document_ids"][:, :-1]

            mask = generate_inter_document_mask(document_ids, num_heads=num_attention_heads)

            if backend == "te":
                if te_fp8:
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        logits = model_engine(input_ids=input_ids, 
                                              self_attn_mask_type = "no_mask",
                                              window_size = (-1, -1),
                                              core_attention_bias_type = 'post_scale_bias',
                                              core_attention_bias = mask
                                              )
                else:
                    logits = model_engine(input_ids=input_ids, 
                                          self_attn_mask_type = "no_mask",
                                          window_size = (-1, -1),
                                          core_attention_bias_type = 'post_scale_bias',
                                          core_attention_bias = mask
                                          )
            else:
                logits = model_engine(input_ids=input_ids, attention_mask=mask)

            if args.train_mode == 'sft':
                loss = sft_loss(data["document_ids"][:, 1:], logits, data["input_ids"][:, 1:])
            else:
                loss = dpo_loss(data["document_ids"][:, 1:], logits, data["input_ids"][:, 1:], args.beta, args.gamma)
        else:
            input_ids = data["input_ids"][:, :-1]
            target = data["input_ids"][:, 1:].contiguous().view(-1)
            
            if te_fp8:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits = model_engine(input_ids=input_ids)
            else:
                logits = model_engine(input_ids=input_ids)

            logits = logits.contiguous().view(-1, logits.shape[-1])
            loss = thunderllm.util.CEloss(logits, target)

        model_engine.backward(loss)
        model_engine.step()

        thunderllm.util.logger.report_train_step(
            step=step,
            max_steps=args.max_steps,
            local_loss=loss.item(),
            local_ce_loss=loss.item() if args.train_mode != 'dpo' else 0.0,
            local_non_ce_loss=0.0 if args.train_mode != 'dpo' else loss.item(),
            model_engine=model_engine,
            input_tokens=data["input_ids"],
            lr_scheduler=lr_scheduler
        )
