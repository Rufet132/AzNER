import json
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

MODEL_NAME = "xlm-roberta-base" 
LABEL_LIST = [
    "O",
    "B-DATE", "I-DATE",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
    "B-PER", "I-PER",
    "B-PRODUCT", "I-PRODUCT"
]
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["ner_tags"] = [label2id[l] for l in item["labels"]]
    return Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=128, 
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_class_weights(train_dataset):
    counts = np.zeros(len(LABEL_LIST))
    for item in train_dataset:
        for lbl_id in item["labels"]:
            if lbl_id != -100:
                counts[lbl_id] += 1
                
    total = np.sum(counts)
    weights = np.power(total / (counts + 1e-5), 0.35)
    
    weights = weights / weights[label2id["O"]]
    weights = np.clip(weights, 0.5, 5.0)
    
    print("\n⚖️ Counted Class Weights:")
    for lbl, w in zip(LABEL_LIST, weights):
        print(f"   {lbl:<12}: {w:.2f}")
        
    return torch.tensor(weights, dtype=torch.float)


class WeightedNERTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

def main():
    print(f"🖥️ GPU aktivdirmi?: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    raw_datasets = DatasetDict({
        "train": load_json_dataset("./data/annotated/train.json"),
        "validation": load_json_dataset("./data/annotated/val.json"),
        "test": load_json_dataset("./data/annotated/test.json"),
    })

    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)
    class_weights = compute_class_weights(tokenized_datasets["train"])

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.25,
        attention_probs_dropout_prob=0.25
    )

    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./ner_model_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,      
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,      
        num_train_epochs=12,               
        weight_decay=0.05,                  
        warmup_steps=40,                    
        fp16=torch.cuda.is_available(),     
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,                 
        logging_steps=20,
        report_to="none"
    )

    trainer = WeightedNERTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\n🚀 Training starts...")
    trainer.train()

    print("\n📈 The last results:")
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    print(classification_report(true_labels, true_predictions))
    
    trainer.save_model("./final_az_ner_model")
    tokenizer.save_pretrained("./final_az_ner_model")
    print("✅ Model saved to './final_az_ner_model'")

if __name__ == "__main__":
    main()
