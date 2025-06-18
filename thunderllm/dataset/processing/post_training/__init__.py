from datasets import load_dataset

# Import processing functions for various datasets
from .gsm8k import process_gsm8k
from .hellaswag import process_hellaswag
from .winogrande import process_winogrande
from .openbookqa import process_obqa
from .arc import process_arc
from .mmlu import process_mmlu
from .kobest_hellaswag import process_kobest_hellaswag
from .kmmlu import process_kmmlu
from .thunder import process_thunder_for_sft, process_thunder_for_dpo

# Define available dataset processors
_DATASET_PROCESSORS = {
    "gsm8k": process_gsm8k,
    "hellaswag": process_hellaswag,
    "winogrande": process_winogrande,
    "openbookqa": process_obqa,
    "arc-e": process_arc,
    "arc-c": process_arc,
    "mmlu": process_mmlu,
    "kobest_hellaswag": process_kobest_hellaswag,
    "kmmlu": process_kmmlu,
}

# Define Thunder dataset processors
_THUNDER_DATASETS = {
    "thunder-instruction": "thunder-research-group/SNU_Thunder-synthetic-instruction-following",
    "thunder-coding": "thunder-research-group/SNU_Thunder-synthetic-coding",
    "thunder-math": "thunder-research-group/SNU_Thunder-synthetic-math",
}


def process_training_set_of(dataset_name: str, **kwargs):
    """
    Process the training set of a benchmark dataset for post-training.
    
    Args:
        dataset_name: Name of the dataset to process.
        **kwargs: Additional arguments to pass to the dataset processor.
    
    Returns:
        Processed dataset in the appropriate format.
    """
    if dataset_name not in _DATASET_PROCESSORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(_DATASET_PROCESSORS.keys())}")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, **kwargs)
    
    # Process and return
    return _DATASET_PROCESSORS[dataset_name](dataset["train"])


def process_thunder_dataset(dataset_name: str, format_type: str = "sft", config: str = "english", **kwargs):
    """
    Process a Thunder dataset for SFT or DPO.
    
    Args:
        dataset_name: Name of the Thunder dataset to process. Can be full name or alias.
        format_type: "sft" or "dpo" depending on the desired format.
        config: Configuration to use (default: "english").
        **kwargs: Additional arguments to pass to the dataset processor.
    
    Returns:
        Processed dataset in the appropriate format.
    """
    # Resolve dataset name if it's an alias
    if dataset_name in _THUNDER_DATASETS:
        full_dataset_name = _THUNDER_DATASETS[dataset_name]
    else:
        full_dataset_name = dataset_name
    
    # Process based on format type
    if format_type.lower() == "sft":
        return process_thunder_for_sft(full_dataset_name, config, **kwargs)
    elif format_type.lower() == "dpo":
        return process_thunder_for_dpo(full_dataset_name, config, **kwargs)
    else:
        raise ValueError(f"Unknown format type: {format_type}. Available: 'sft', 'dpo'")


__all__ = [
    "process_training_set_of",
    "process_thunder_dataset",
]
