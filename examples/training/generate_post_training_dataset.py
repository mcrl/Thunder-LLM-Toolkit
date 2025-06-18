#!/usr/bin/env python3
import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
from typing import List, Dict, Tuple, Any, Optional, Union
from datasets import Dataset
from thunderllm.dataset.processing.post_training import process_thunder_dataset, process_training_set_of

def parse_args():
    parser = argparse.ArgumentParser(description="Generate packed dataset from Thunder datasets for training")
    
    parser.add_argument("--task", type=str, choices=["sft", "dpo"], required=True,
                        help="Task type: 'sft' for supervised fine-tuning or 'dpo' for direct preference optimization")
    
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer model")
    
    parser.add_argument("--synthetic-dataset_names", type=str, nargs="+", required=True,
                        help="Names of Thunder datasets to process (e.g., thunder-research-group/SNU_Thunder-synthetic-math)")
    parser.add_argument("--dataset_names", type=str, nargs="+", required=True,
                        help="Names of datasets to process (e.g., hellaswag)")
    
    parser.add_argument("--output_dir", type=str, default="packed_dataset",
                        help="Output directory for the packed parquet files")
    
    parser.add_argument("--config", type=str, default="english",
                        help="Dataset configuration to use")
    
    parser.add_argument("--max_seq_length", type=int, default=8193,
                        help="Maximum sequence length for packed sequences")
    
    parser.add_argument("--num_files", type=int, default=32,
                        help="Number of output parquet files to create")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling and sampling")
    
    return parser.parse_args()

def process_sft_dataset(args):
    """Process datasets for Supervised Fine-Tuning (SFT)"""
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Loaded tokenizer from {args.tokenizer_path}")
    
    all_input_ids = []
    all_question_length = []
    
    # Process synthetic datasets using process_thunder_dataset
    if hasattr(args, 'synthetic_dataset_names') and args.synthetic_dataset_names:
        for dataset_name in tqdm(args.synthetic_dataset_names, desc="Processing synthetic datasets"):
            try:
                print(f"Processing synthetic dataset {dataset_name} for SFT")
                
                # Use Thunder processing function to get the dataset
                dataset = process_thunder_dataset(
                    dataset_name=dataset_name,
                    task="sft",
                    config=args.config
                )
                
                # Convert to list for processing
                data = dataset.to_pandas().to_dict('records')
                print(f"Got {len(data)} examples from {dataset_name}")
                
                for ele in tqdm(data, desc=f"Processing {dataset_name}", leave=False):
                    question = ele["question"]
                    answer = ele["correct"]  # From Thunder processing

                    # Add space between question and answer if needed
                    needs_delimiter = (not question.endswith(" ") and 
                                      not question.endswith("\n") and 
                                      not answer.startswith(" ") and 
                                      not answer.startswith("\n"))
                    if needs_delimiter:
                        question += " "

                    full_text = question + answer
                    question_only = question
                    
                    # Tokenize text
                    full_encoded = tokenizer(full_text, truncation=True, max_length=args.max_seq_length)
                    question_encoded = tokenizer(question_only, truncation=True, max_length=args.max_seq_length)
                    
                    question_length = len(question_encoded['input_ids'])
                    
                    all_input_ids.append(full_encoded['input_ids'])
                    all_question_length.append(question_length)
                
            except Exception as e:
                print(f"Error processing synthetic dataset {dataset_name} for SFT: {e}")
    
    # Process regular datasets using process_training_set_of
    if hasattr(args, 'dataset_names') and args.dataset_names:
        for dataset_name in tqdm(args.dataset_names, desc="Processing regular datasets"):
            try:
                print(f"Processing regular dataset {dataset_name} for SFT")
                
                # Use process_training_set_of function to get the dataset
                dataset = process_training_set_of(
                    dataset_name=dataset_name
                )
                
                # Convert to list for processing
                data = dataset.to_pandas().to_dict('records')
                print(f"Got {len(data)} examples from {dataset_name}")
                
                for ele in tqdm(data, desc=f"Processing {dataset_name}", leave=False):
                    # Handle different key structures
                    if "question" in ele and ("correct" in ele or "answer" in ele):
                        question = ele["question"]
                        answer = ele.get("correct", ele.get("answer", ""))
                    else:
                        print(f"Warning: Unexpected data format in {dataset_name}. Skipping...")
                        continue

                    # Add space between question and answer if needed
                    needs_delimiter = (not question.endswith(" ") and 
                                      not question.endswith("\n") and 
                                      not answer.startswith(" ") and 
                                      not answer.startswith("\n"))
                    if needs_delimiter:
                        question += " "

                    full_text = question + answer
                    question_only = question
                    
                    # Tokenize text
                    full_encoded = tokenizer(full_text, truncation=True, max_length=args.max_seq_length)
                    question_encoded = tokenizer(question_only, truncation=True, max_length=args.max_seq_length)
                    
                    question_length = len(question_encoded['input_ids'])
                    
                    all_input_ids.append(full_encoded['input_ids'])
                    all_question_length.append(question_length)
                
            except Exception as e:
                print(f"Error processing regular dataset {dataset_name} for SFT: {e}")
    
    # Shuffle the data
    indices = np.random.permutation(len(all_input_ids))
    shuffled_input_ids = [all_input_ids[i] for i in indices]
    shuffled_question_length = [all_question_length[i] for i in indices]

    return shuffled_input_ids, shuffled_question_length, tokenizer

def process_dpo_dataset(args):
    """Process datasets for Direct Preference Optimization (DPO)"""
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Loaded tokenizer from {args.tokenizer_path}")
    
    all_input_ids = []
    all_question_length = []
    
    # Process synthetic datasets using process_thunder_dataset
    if hasattr(args, 'synthetic_dataset_names') and args.synthetic_dataset_names:
        for dataset_name in tqdm(args.synthetic_dataset_names, desc="Processing synthetic datasets"):
            try:
                print(f"Processing synthetic dataset {dataset_name} for DPO")
                
                # Use Thunder processing function to get the dataset
                dataset = process_thunder_dataset(
                    dataset_name=dataset_name,
                    task="dpo",
                    config=args.config
                )
                
                # Convert to list for processing
                data = dataset.to_pandas().to_dict('records')
                print(f"Got {len(data)} pairs from {dataset_name}")
                
                for ele in tqdm(data, desc=f"Processing {dataset_name}", leave=False):
                    # Get question from conversations
                    if "conversations" in ele and len(ele["conversations"]) > 0:
                        question = ele["conversations"][0]["value"]
                    else:
                        continue

                    # Get chosen and rejected responses
                    if "chosen" in ele and "rejected" in ele:
                        if isinstance(ele["chosen"], dict) and "value" in ele["chosen"]:
                            chosen = ele["chosen"]["value"]
                        else:
                            continue
                            
                        if isinstance(ele["rejected"], dict) and "value" in ele["rejected"]:
                            rejected = ele["rejected"]["value"]
                        else:
                            continue
                    else:
                        continue

                    # Add space between question and responses if needed
                    chosen_delimiter_needed = (not question.endswith(" ") and 
                                              not question.endswith("\n") and 
                                              not chosen.startswith(" ") and 
                                              not chosen.startswith("\n"))
                    if chosen_delimiter_needed:
                        chosen = " " + chosen
                    
                    rejected_delimiter_needed = (not question.endswith(" ") and
                                               not question.endswith("\n") and
                                               not rejected.startswith(" ") and
                                               not rejected.startswith("\n"))
                    if rejected_delimiter_needed:
                        rejected = " " + rejected

                    chosen_full = question + chosen
                    rejected_full = question + rejected
                    question_only = question
                    
                    # Tokenize
                    chosen_encoded = tokenizer(chosen_full, truncation=True, max_length=args.max_seq_length)
                    rejected_encoded = tokenizer(rejected_full, truncation=True, max_length=args.max_seq_length)
                    question_encoded = tokenizer(question_only, truncation=True, max_length=args.max_seq_length)
                    
                    question_length = len(question_encoded['input_ids'])
                    
                    all_input_ids.append((chosen_encoded['input_ids'], rejected_encoded['input_ids']))
                    all_question_length.append(question_length)
            
            except Exception as e:
                print(f"Error processing synthetic dataset {dataset_name} for DPO: {e}")
    
    # Process regular datasets using process_training_set_of
    if hasattr(args, 'dataset_names') and args.dataset_names:
        for dataset_name in tqdm(args.dataset_names, desc="Processing regular datasets"):
            try:
                print(f"Processing regular dataset {dataset_name} for DPO")
                
                # Use process_training_set_of function to get the dataset
                dataset = process_training_set_of(
                    dataset_name=dataset_name
                )
                
                # Convert to list for processing
                data = dataset.to_pandas().to_dict('records')
                print(f"Got {len(data)} pairs from {dataset_name}")
                
                for ele in tqdm(data, desc=f"Processing {dataset_name}", leave=False):
                    # Get question (could be in different formats)
                    if "conversations" in ele and len(ele["conversations"]) > 0:
                        question = ele["conversations"][0]["value"]
                    elif "question" in ele:
                        question = ele["question"]
                    else:
                        continue

                    # Get chosen and rejected responses
                    if "chosen" in ele and "rejected" in ele:
                        if isinstance(ele["chosen"], dict) and "value" in ele["chosen"]:
                            chosen = ele["chosen"]["value"]
                        else:
                            chosen = ele["chosen"]
                            
                        if isinstance(ele["rejected"], dict) and "value" in ele["rejected"]:
                            rejected = ele["rejected"]["value"]
                        else:
                            rejected = ele["rejected"]
                    else:
                        continue

                    # Add space between question and responses if needed
                    chosen_delimiter_needed = (not question.endswith(" ") and 
                                              not question.endswith("\n") and 
                                              not chosen.startswith(" ") and 
                                              not chosen.startswith("\n"))
                    if chosen_delimiter_needed:
                        chosen = " " + chosen
                    
                    rejected_delimiter_needed = (not question.endswith(" ") and
                                               not question.endswith("\n") and
                                               not rejected.startswith(" ") and
                                               not rejected.startswith("\n"))
                    if rejected_delimiter_needed:
                        rejected = " " + rejected

                    chosen_full = question + chosen
                    rejected_full = question + rejected
                    question_only = question
                    
                    # Tokenize
                    chosen_encoded = tokenizer(chosen_full, truncation=True, max_length=args.max_seq_length)
                    rejected_encoded = tokenizer(rejected_full, truncation=True, max_length=args.max_seq_length)
                    question_encoded = tokenizer(question_only, truncation=True, max_length=args.max_seq_length)
                    
                    question_length = len(question_encoded['input_ids'])
                    
                    all_input_ids.append((chosen_encoded['input_ids'], rejected_encoded['input_ids']))
                    all_question_length.append(question_length)
            
            except Exception as e:
                print(f"Error processing regular dataset {dataset_name} for DPO: {e}")
    
    # Shuffle the data
    indices = np.random.permutation(len(all_input_ids))
    shuffled_input_ids = [all_input_ids[i] for i in indices]
    shuffled_question_length = [all_question_length[i] for i in indices]

    return shuffled_input_ids, shuffled_question_length, tokenizer

def create_packed_sequences_sft(all_input_ids, all_question_length, tokenizer, max_seq_length=8193):
    """Create packed sequences for SFT data"""
    
    packed_input_ids = []
    packed_document_ids = []
    
    current_input_ids = []
    current_document_ids = []
    current_doc_id = 1  # Start with document ID 1 (0 is reserved for padding)
    
    print("Creating packed sequences...")

    for idx in range(len(all_input_ids)):
        sample_ids = all_input_ids[idx]
        sample_length = len(sample_ids)
        question_length = all_question_length[idx]
        
        # Skip if this single sample is too long
        if sample_length > max_seq_length:
            print(f"Skipping sample with length {sample_length} (exceeds max_seq_length)")
            continue
            
        # Start a new sequence if the current one would overflow
        if current_input_ids and (len(current_input_ids) + sample_length > max_seq_length):
            # Add padding to complete the sequence
            padding_length = max_seq_length - len(current_input_ids)
            if padding_length > 0:
                current_input_ids.extend([tokenizer.eos_token_id] * padding_length)
                current_document_ids.extend([0] * padding_length)
                
            packed_input_ids.append(current_input_ids)
            packed_document_ids.append(current_document_ids)
            
            current_input_ids = []
            current_document_ids = []
            current_doc_id = 1
        
        # Add the sample to the current sequence
        current_input_ids.extend(sample_ids)
        # Add document IDs: one for question part, another for answer part
        current_document_ids.extend([current_doc_id] * question_length)
        current_doc_id += 1
        current_document_ids.extend([current_doc_id] * (sample_length - question_length))
        current_doc_id += 1
    
    # Handle the last sequence if it's not empty
    if current_input_ids:
        padding_length = max_seq_length - len(current_input_ids)
        if padding_length > 0:
            current_input_ids.extend([tokenizer.eos_token_id] * padding_length)
            current_document_ids.extend([0] * padding_length)  # 0 for padding document ID
            
        packed_input_ids.append(current_input_ids)
        packed_document_ids.append(current_document_ids)
    
    print(f"Created {len(packed_input_ids)} packed sequences from {len(all_input_ids)} original samples")
    
    # Calculate packing efficiency
    total_tokens = sum(len(ids) for ids in all_input_ids)
    max_possible_tokens = len(packed_input_ids) * max_seq_length
    packing_efficiency = total_tokens / max_possible_tokens if max_possible_tokens > 0 else 0
    print(f"Packing efficiency: {packing_efficiency:.2f}")
    
    return packed_input_ids, packed_document_ids

def create_packed_sequences_dpo(all_input_ids, all_question_length, tokenizer, max_seq_length=8193):
    """Create packed sequences for DPO data"""
    
    packed_input_ids = []
    packed_document_ids = []
    
    current_input_ids = []
    current_document_ids = []
    current_doc_id = 1  # Start with document ID 1 (0 is reserved for padding)
    
    print("Creating packed sequences...")

    for idx in range(len(all_input_ids)):
        chosen_ids, rejected_ids = all_input_ids[idx]
        chosen_length = len(chosen_ids)
        rejected_length = len(rejected_ids)
        question_length = all_question_length[idx]

        sample_length = chosen_length + rejected_length
        
        # Skip if this single sample is too long
        if sample_length > max_seq_length:
            print(f"Skipping sample with length {sample_length} (exceeds max_seq_length)")
            continue
            
        # Start a new sequence if the current one would overflow
        if current_input_ids and (len(current_input_ids) + sample_length > max_seq_length):
            # Add padding to complete the sequence
            padding_length = max_seq_length - len(current_input_ids)
            if padding_length > 0:
                current_input_ids.extend([tokenizer.eos_token_id] * padding_length)
                current_document_ids.extend([0] * padding_length)
                
            packed_input_ids.append(current_input_ids)
            packed_document_ids.append(current_document_ids)
            
            current_input_ids = []
            current_document_ids = []
            current_doc_id = 1
        
        # Add the chosen response
        current_input_ids.extend(chosen_ids)
        current_document_ids.extend([current_doc_id] * question_length)
        current_doc_id += 1
        current_document_ids.extend([current_doc_id] * (chosen_length - question_length))
        current_doc_id += 1

        # Add the rejected response
        current_input_ids.extend(rejected_ids)
        current_document_ids.extend([current_doc_id] * question_length)
        current_doc_id += 1
        current_document_ids.extend([current_doc_id] * (rejected_length - question_length))
        current_doc_id += 1
    
    # Handle the last sequence if it's not empty
    if current_input_ids:
        padding_length = max_seq_length - len(current_input_ids)
        if padding_length > 0:
            current_input_ids.extend([tokenizer.eos_token_id] * padding_length)
            current_document_ids.extend([0] * padding_length)  # 0 for padding document ID
            
        packed_input_ids.append(current_input_ids)
        packed_document_ids.append(current_document_ids)
    
    print(f"Created {len(packed_input_ids)} packed sequences from {len(all_input_ids)} original pairs")
    
    # Calculate packing efficiency (each element in all_input_ids is a tuple with chosen and rejected)
    total_tokens = sum(len(chosen) + len(rejected) for chosen, rejected in all_input_ids)
    max_possible_tokens = len(packed_input_ids) * max_seq_length
    packing_efficiency = total_tokens / max_possible_tokens if max_possible_tokens > 0 else 0
    print(f"Packing efficiency: {packing_efficiency:.2f}")
    
    return packed_input_ids, packed_document_ids

def save_to_parquet_files(packed_input_ids, packed_document_ids, output_dir, tokenizer_path, max_seq_length, num_files=32):
    """Save packed sequences to parquet files and create meta_info.json"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = len(packed_document_ids)
    samples_per_file = max(1, total_samples // num_files)
    
    print(f"Total samples: {total_samples}, Samples per file: {samples_per_file}")
    
    for i in range(num_files):
        start_idx = i * samples_per_file
        end_idx = min((i + 1) * samples_per_file, total_samples)
        
        if start_idx >= total_samples:
            break
        
        current_input_ids = packed_input_ids[start_idx:end_idx]
        current_document_ids = packed_document_ids[start_idx:end_idx]
        
        df = pd.DataFrame({
            'input_ids': current_input_ids,
            'document_ids': current_document_ids
        })
        
        input_ids_type = pa.list_(pa.int32())
        document_ids_type = pa.list_(pa.int32())
        
        table = pa.Table.from_arrays(
            [pa.array(df['input_ids'], type=input_ids_type),
             pa.array(df['document_ids'], type=document_ids_type)],
            ['input_ids', 'document_ids']
        )
        
        file_name = f"{i:08d}.parquet"
        output_path = os.path.join(output_dir, file_name)
        
        pq.write_table(table, output_path)
        print(f"Saved file {i+1}/{num_files}: {output_path} with {end_idx - start_idx} samples")
    
    # Create meta_info.json file
    meta_info = {
        "total_samples": total_samples,
        "num_files": num_files,
        "samples_per_file": samples_per_file,
        "file_format": "parquet",
        "tokenizer_name_or_path": tokenizer_path,
        "block_size": max_seq_length
    }
    
    meta_info_path = os.path.join(output_dir, "meta_info.json")
    with open(meta_info_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)
    
    print(f"Created meta_info.json with dataset information at {meta_info_path}")

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Starting data processing for {args.task.upper()} task...")
    
    if args.task == "sft":
        all_input_ids, all_question_length, tokenizer = process_sft_dataset(args)
        packed_input_ids, packed_document_ids = create_packed_sequences_sft(
            all_input_ids, all_question_length, tokenizer, max_seq_length=args.max_seq_length
        )
    else:  # dpo
        all_input_ids, all_question_length, tokenizer = process_dpo_dataset(args)
        packed_input_ids, packed_document_ids = create_packed_sequences_dpo(
            all_input_ids, all_question_length, tokenizer, max_seq_length=args.max_seq_length
        )
    
    print(f"Processed {len(all_input_ids)} samples.")
    
    # Create output directory with task name
    output_dir = os.path.join(args.output_dir, args.task)
    save_to_parquet_files(
        packed_input_ids, 
        packed_document_ids, 
        output_dir, 
        args.tokenizer_path,
        args.max_seq_length,
        num_files=args.num_files
    )
    
    print(f"Processing complete! Dataset saved to {output_dir}")

if __name__ == "__main__":
    main()
