import json
import re
import random
from collections import Counter


def load_label_studio_json(file_path):
    """Loads a Label Studio JSON export file in UTF-8 encoding."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_tokens_with_regex(text):
    """Tokenizes text into words (supporting Azerbaijani characters) and punctuation with character offsets."""
    tokens_info = []
    for m in re.finditer(r'\w+|[^\w\s]', text):
        tokens_info.append({
            'word': m.group(),
            'start': m.start(),
            'end': m.end()
        })
    return tokens_info


def align_tokens_and_spans(text, spans):
    """Aligns regex-extracted tokens with character-level entity spans and converts them to BIO format."""
    tokens_list = []
    labels_list = []
    used_spans = set()
    
    sorted_spans = sorted(spans, key=lambda x: x['start'])
    tokens_info = extract_tokens_with_regex(text)
    
    for token in tokens_info:
        token_word = token['word']
        token_start = token['start']
        token_end = token['end']
        
        assigned_label = "O"
        
        for s in sorted_spans:
            # Check for mathematical intersection between token and entity span
            if token_start < s['end'] and token_end > s['start']:
                span_id = (s['start'], s['end'])
                
                if span_id not in used_spans:
                    assigned_label = f"B-{s['label']}"
                    used_spans.add(span_id)
                else:
                    assigned_label = f"I-{s['label']}"
                
                break
                
        tokens_list.append(token_word)
        labels_list.append(assigned_label)
        
    return tokens_list, labels_list


def process_dataset(raw_data):
    """Processes raw Label Studio items into tokenized BIO sequences and detects overlapping spans."""
    processed_data = []
    overlap_list = []
    
    for item in raw_data:
        sentence_id = item['id']
        text = item['data']['text']
        spans = []
        
        if 'annotations' in item and len(item['annotations']) > 0:
            annotation = item['annotations'][0]
            if 'result' in annotation:
                for res in annotation['result']:
                    if 'value' in res and 'labels' in res['value']:
                        val = res['value']
                        spans.append({
                            'start': val['start'],
                            'end': val['end'],
                            'label': val['labels'][0]
                        })
        
        # Detect nested/overlapping spans
        sorted_spans = sorted(spans, key=lambda x: x['start'])
        for i in range(len(sorted_spans) - 1):
            if sorted_spans[i]['end'] > sorted_spans[i+1]['start']:
                overlap_list.append({
                    'task_id': sentence_id,
                    'span1': sorted_spans[i],
                    'span2': sorted_spans[i+1]
                })
        
        tokens, labels = align_tokens_and_spans(text, spans)
        processed_data.append({
            'id': sentence_id,
            'tokens': tokens,
            'labels': labels
        })
        
    if overlap_list:
        print(f"⚠️ WARNING: {len(overlap_list)} overlapping spans detected.")
        for ov in overlap_list:
            print(f"   Task ID: {ov['task_id']}")
            print(f"   Span 1: {ov['span1']}")
            print(f"   Span 2: {ov['span2']}\n")

    return processed_data


def run_validations(processed_data, raw_spans_count):
    """Validates BIO tag counts, sequence transitions, and prints random samples for inspection."""
    print("\n--- VALIDATION REPORT ---\n")
    
    # (a) Verify total 'B-' tags match expected raw span count
    total_b_count = 0
    for item in processed_data:
        for label in item['labels']:
            if label.startswith("B-"):
                total_b_count += 1
                
    print(f"(a) Total 'B-' tags found: {total_b_count}")
    print(f"    Expected raw spans: {raw_spans_count}")
    if total_b_count == raw_spans_count:
        print("    ✅ PASSED: 'B-' tag count matches total span count.")
    else:
        print("    ❌ FAILED: Tag count mismatch! Please inspect the alignment logic.")
        
    # (b) Verify 'I-X' tags only follow 'B-X' or 'I-X'
    invalid_i_sequence = False
    for item in processed_data:
        labels = item['labels']
        for i in range(len(labels)):
            if labels[i].startswith("I-"):
                if i == 0:
                    invalid_i_sequence = True
                    break
                prev_label = labels[i-1]
                current_entity = labels[i].split("-")[1]
                
                if prev_label == "O":
                    invalid_i_sequence = True
                    break
                else:
                    prev_entity = prev_label.split("-")[1]
                    if prev_entity != current_entity:
                        invalid_i_sequence = True
                        break
                        
    if not invalid_i_sequence:
        print("(b) ✅ PASSED: All 'I-' tags follow valid BIO sequence rules.")
    else:
        print("(b) ❌ FAILED: Invalid 'I-' tag transition detected!")

    # (c) Print 5 random samples for manual visual inspection
    print("\n(c) RANDOM SAMPLE INSPECTION (5 SENTENCES):\n")
    random_samples = random.sample(processed_data, min(5, len(processed_data)))
    
    for idx, sample in enumerate(random_samples, 1):
        print(f"Sample {idx} (Sentence ID: {sample['id']}):")
        print("-" * 50)
        for t, l in zip(sample['tokens'], sample['labels']):
            if l != "O":
                print(f"  {t:<20} -> {l}")
            else:
                print(f"  {t:<20} -> {l}")
        print("=" * 50 + "\n")


def split_dataset(processed_data):
    """
    Splits the dataset sequentially into Train (70%), Val (15%), and Test (15%).
    Prints label distribution for each split using optimized Counter.
    """
    total_count = len(processed_data)
    
    train_end = int(total_count * 0.70)
    val_end = train_end + int(total_count * 0.15)
    
    splits = {
        "TRAIN": processed_data[:train_end],
        "VAL": processed_data[train_end:val_end],
        "TEST": processed_data[val_end:]
    }

    
    for name, data in splits.items():
        # Optimized: Flatten all labels and filter 'B-' in a single step using Counter
        label_counts = Counter(
            lbl for item in data for lbl in item['labels'] if lbl.startswith("B-")
        )
                    
        print(f"\n--- {name} Split Statistics (Total Sentences: {len(data)}) ---")
        for lbl, count in label_counts.items():
            print(f"  {lbl}: {count}")
            if name == "TEST" and count < 5:
                print(f"  ⚠️ WARNING: {lbl} has less than 5 entities in TEST split. Results might be unreliable.")
                
    return splits["TRAIN"], splits["VAL"], splits["TEST"]


# ==========================================
# BAŞLADICI (MAIN EXECUTION) BLOCK
# ==========================================
if __name__ == "__main__":
    json_file_path = "./data/annotated/data_all.json" 
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_json_data = json.load(f)
            
        real_json_spans_count = 0
        for item in raw_json_data:
            if 'annotations' in item and len(item['annotations']) > 0:
                for res in item['annotations'][0].get('result', []):
                    if 'value' in res and 'labels' in res['value']:
                        real_json_spans_count += 1
                        
        print(f"📊 ZƏMANƏT TESTİ: JSON-un içində tapılan real etiket (span) sayı: {real_json_spans_count}")
        
        final_dataset = process_dataset(raw_json_data)
        
        run_validations(final_dataset, raw_spans_count=real_json_spans_count)
        
    except FileNotFoundError:
        print(f"❌ XƏTA: '{json_file_path}' faylı tapılmadı.")



    try:
        raw_json_data = load_label_studio_json(json_file_path)
        print(f"File loaded successfully. Total sentences: {len(raw_json_data)}")
        
        final_dataset = process_dataset(raw_json_data)
        
        # New Step: Split the dataset and print statistics
        train_set, val_set, test_set = split_dataset(final_dataset)
        
    except FileNotFoundError:
        print(f"❌ ERROR: '{json_file_path}' not found.")
