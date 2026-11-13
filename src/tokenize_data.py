import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from convert_to_bio import load_label_studio_json, process_dataset, split_dataset

# 1. Define the exact 11 labels and their ID mappings
unique_labels = [
    "O",
    "B-LOC", "I-LOC",
    "B-PER", "I-PER",
    "B-DATE", "I-DATE",
    "B-ORG", "I-ORG",
    "B-PRODUCT", "I-PRODUCT"
]

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for idx, label in enumerate(unique_labels)}

def tokenize_and_align_batch(examples, tokenizer, max_length=128):
    """
    Processes a batch of samples for Hugging Face Dataset mapping.
    Converts string labels into numerical label IDs with -100 masking.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length
    )
    
    batch_all_labels = []
    
    for i, labels_list in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                label_str = labels_list[word_idx]
                aligned_labels.append(label2id[label_str])
            else:
                aligned_labels.append(-100)
                
            previous_word_idx = word_idx
            
        batch_all_labels.append(aligned_labels)
        
    tokenized_inputs["labels"] = batch_all_labels
    return tokenized_inputs

# --- GLOBAL EXECUTION FOR TRAINING IMPORT ---
# Load and preprocessing raw data (Make sure label_studio_data.json is in /content/)
raw_data = load_label_studio_json("all_data.json")
final_dataset = process_dataset(raw_data)

# Split sequentially (70/15/15)
train_data, val_data, test_data = split_dataset(final_dataset)

# Convert to Hugging Face Dataset format
hf_dict = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data)
})

# Load the tokenizer global instance
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Tokenize all splits using .map()
encoded_dataset = hf_dict.map(
    lambda x: tokenize_and_align_batch(x, tokenizer, max_length=128),
    batched=True
)
