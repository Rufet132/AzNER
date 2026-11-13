import json
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report

MODEL_PATH = "./final_az_ner_model"
LABEL_LIST = ["O", "B-DATE", "I-DATE", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-PRODUCT", "I-PRODUCT"]
label2id = {l: i for i, l in enumerate(LABEL_LIST)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

with open("./data/annotated/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
for item in test_data:
    item["ner_tags"] = [label2id[l] for l in item["labels"]]

test_ds = Dataset.from_list(test_data)

def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, max_length=128, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        lbl_ids = []
        for w_id in word_ids:
            if w_id is None or w_id == prev:
                lbl_ids.append(-100)
            else:
                lbl_ids.append(label[w_id])
            prev = w_id
        labels.append(lbl_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_test = test_ds.map(tokenize_and_align, batched=True)
trainer = Trainer(model=model, data_collator=DataCollatorForTokenClassification(tokenizer))

preds, labels, _ = trainer.predict(tokenized_test)
preds = np.argmax(preds, axis=2)

true_preds = [[LABEL_LIST[p] for p, l in zip(pr, lb) if l != -100] for pr, lb in zip(preds, labels)]
true_labels = [[LABEL_LIST[l] for p, l in zip(pr, lb) if l != -100] for pr, lb in zip(preds, labels)]

report_str = classification_report(true_labels, true_preds, digits=4)
report_dict = classification_report(true_labels, true_preds, output_dict=True)

print(report_str)

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=4)
print("✅ Detallı nəticələr 'evaluation_results.json' faylına yazıldı!")
