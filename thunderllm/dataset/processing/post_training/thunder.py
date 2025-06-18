import os
import random
import json
from typing import List, Dict, Any, Optional, Union
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm


def process_thunder_for_sft(dataset_name: str, config: str = "english") -> Dataset:
    """
    Process Thunder datasets for Supervised Fine-Tuning (SFT).
    Only uses examples with chosen=True.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "thunder-research-group/SNU_Thunder-synthetic-math")
        config: Configuration to use (default: "english")

    Returns:
        Huggingface Dataset with formatted SFT data
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, config)
    
    # Convert to DataFrame for processing
    df = pd.DataFrame(dataset)
    
    # Extract the nested dictionaries
    nested_data = []
    for _, row in df.iterrows():
        nested_data.append(row[df.columns[0]])
    df = pd.DataFrame(nested_data)

    # Filter chosen examples only
    if "chosen" in df.columns:
        sft_df = df[df["chosen"] == True].copy()
    else:
        raise ValueError(f"Dataset {dataset_name} does not have a 'chosen' column")
    
    # Format for SFT
    sft_data = []
    
    for _, row in tqdm(sft_df.iterrows(), total=len(sft_df), desc="Processing for SFT"):
        sft_item = {
            "question": row["question"],
            "correct": row["response"]
        }
        sft_data.append(sft_item)
    
    # Convert to Dataset
    return Dataset.from_pandas(pd.DataFrame(sft_data))


def process_thunder_for_dpo(dataset_name: str, config: str = "english") -> Dataset:
    """
    Process Thunder datasets for Direct Preference Optimization (DPO).
    Creates pairs with one chosen and one rejected response per question.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "thunder-research-group/SNU_Thunder-synthetic-math")
        config: Configuration to use (default: "english")
        
    Returns:
        Huggingface Dataset with DPO formatted data
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, config)

    # Convert to DataFrame for processing
    df = pd.DataFrame(dataset)

    # Extract the nested dictionaries
    nested_data = []
    for _, row in df.iterrows():
        nested_data.append(row[df.columns[0]])
    df = pd.DataFrame(nested_data)
    
    # Check if dataset has the required format
    if "chosen" not in df.columns:
        raise ValueError(f"Dataset {dataset_name} does not have a 'chosen' column")
    
    # Group by question to create pairs
    question_groups = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grouping examples"):
        question = row["question"]
        
        if question not in question_groups:
            question_groups[question] = {
                "question": question,
                "chosen_responses": [],
                "rejected_responses": [],
                "question_source": row.get("question_source", ""),
                "source_dataset": row.get("source_dataset", "")
            }
        
        if row["chosen"]:
            question_groups[question]["chosen_responses"].append(row["response"])
        else:
            question_groups[question]["rejected_responses"].append(row["response"])
    
    # Create DPO format
    dpo_data = []
    
    for question_key, group in tqdm(question_groups.items(), desc="Creating DPO pairs"):
        # Skip if we don't have both chosen and rejected responses
        if not group["chosen_responses"] or not group["rejected_responses"]:
            continue
        
        # Randomly sample one chosen and one rejected response
        chosen_response = random.choice(group["chosen_responses"])
        rejected_response = random.choice(group["rejected_responses"])
        
        dpo_item = {
            "question": group["question"],
            "chosen": chosen_response,
            "rejected": rejected_response
        }
        dpo_data.append(dpo_item)
    
    # Convert to Dataset
    return Dataset.from_pandas(pd.DataFrame(dpo_data))