"""
Data loading utilities for DeepSpeed training.

This module handles dataset loading, preprocessing, and dataloader creation
for training language models with DeepSpeed.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """
    Custom Dataset class for WikiText data.
    
    This class preprocesses and wraps the WikiText dataset for PyTorch training.
    It tokenizes the text and prepares input_ids, attention_mask, and labels.
    """
    
    def __init__(self, dataset, preprocess_fn):
        """
        Initialize the dataset.
        
        Args:
            dataset: Raw dataset from Hugging Face datasets
            preprocess_fn: Function to preprocess the examples
        """
        # Use the dataset's map method to apply preprocessing to all examples
        self.tokenized_data = dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=dataset.column_names
        )
        # Set format to PyTorch tensors
        self.tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        """
        Get an item at the specified index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        return self.tokenized_data[idx]


def get_tokenizer(model_name="distilgpt2"):
    """
    Load and configure the tokenizer.
    
    Args:
        model_name: Name of the pretrained model/tokenizer
        
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token to eos_token for GPT-2 models
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def preprocess_function(examples, tokenizer, max_length=128):
    """
    Preprocess text examples into tokenized format.
    
    Args:
        examples: Dictionary containing 'text' field (batch of examples)
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized data including labels
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    
    # For language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def get_dataloaders(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer=None,
    batch_size=8,
    max_length=128,
    num_workers=0
):
    """
    Load and prepare dataloaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of the dataset to load
        dataset_config: Configuration of the dataset
        tokenizer: Tokenizer to use (if None, will load default)
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, eval_loader, test_loader, tokenizer)
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Create preprocess function with tokenizer
    preprocess_fn = lambda x: preprocess_function(x, tokenizer, max_length)
    
    # Create dataset objects
    train_dataset = WikiTextDataset(dataset["train"], preprocess_fn)
    eval_dataset = WikiTextDataset(dataset["validation"], preprocess_fn)
    test_dataset = WikiTextDataset(dataset["test"], preprocess_fn)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, eval_loader, test_loader, tokenizer
