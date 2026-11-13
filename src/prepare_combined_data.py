import json
import random
from collections import Counter
from src.convert_to_bio import load_label_studio_json, process_dataset, run_validations

def normalize_text(tokens):
    return " ".join(tokens).lower().strip()

def check_label_consistency(processed_data):
    unique_labels = set()
    for item in processed_data:
        unique_labels.update(item['labels'])
    print(f"📌 Tapılan bütün unikal BIO etiketləri: {sorted(list(unique_labels))}")
    return unique_labels

def analyze_class_imbalance(processed_data, title="ÜMUMİ DATASET"):
    print(f"\n📊 --- {title} ANALİZİ ---")
    print(f"Cümlə sayı: {len(processed_data)}")
    
    entity_counts = Counter()
    token_counts = Counter()
    sentences_with_product = 0
    
    for item in processed_data:
        has_prod = False
        for label in item['labels']:
            token_counts[label] += 1
            if label.startswith("B-"):
                ent_type = label.split("-")[1]
                entity_counts[ent_type] += 1
                if ent_type.upper() == "PRODUCT":
                    has_prod = True
        if has_prod:
            sentences_with_product += 1

    print("\nEntity Sayları (B- tag üzrə):")
    total_entities = sum(entity_counts.values())
    for ent, count in entity_counts.most_common():
        pct = (count / total_entities * 100) if total_entities > 0 else 0
        print(f"  - {ent:<15}: {count:<5} ({pct:.1f}%)")
        
    print(f"\n⚠️ İçində PRODUCT olan cümlə sayı: {sentences_with_product} / {len(processed_data)}")
    return entity_counts, token_counts

def stratified_split_for_ner(processed_data, train_ratio=0.75, val_ratio=0.15, seed=42):
    random.seed(seed)
    
    product_sents = []
    other_sents = []
    
    for item in processed_data:
        if any("PRODUCT" in lbl.upper() for lbl in item['labels']):
            product_sents.append(item)
        else:
            other_sents.append(item)
            
    random.shuffle(product_sents)
    random.shuffle(other_sents)
    
    def split_list(lst):
        n = len(lst)
        t_end = int(n * train_ratio)
        v_end = t_end + int(n * val_ratio)
        return lst[:t_end], lst[t_end:v_end], lst[v_end:]
        
    p_train, p_val, p_test = split_list(product_sents)
    o_train, o_val, o_test = split_list(other_sents)
    
    splits = {
        "train": p_train + o_train,
        "val": p_val + o_val,
        "test": p_test + o_test
    }
    
    for k in splits:
        random.shuffle(splits[k])
        
    return splits

def main():
    old_data = load_label_studio_json("./data/annotated/data_all.json")
    new_data = load_label_studio_json("./data/annotated/project-10.json")
    
    print("🔄 Köhnə dataset emal olunur...")
    proc_old = process_dataset(old_data)
    
    print("🔄 Yeni dataset emal olunur...")
    proc_new = process_dataset(new_data)
    
    for item in proc_old:
        item['id'] = f"old_{item['id']}"
    for item in proc_new:
        item['id'] = f"new_{item['id']}"
        
    combined_raw = proc_old + proc_new
    
    seen_texts = set()
    clean_data = []
    duplicates_removed = 0
    
    for item in combined_raw:
        if not item['tokens']:
            continue
            
        norm_txt = normalize_text(item['tokens'])
        if norm_txt not in seen_texts:
            seen_texts.add(norm_txt)
            clean_data.append(item)
        else:
            duplicates_removed += 1
            
    print(f"\n🧹 Təmizlik Hesabatı:")
    print(f"   İlkin cümlə sayı: {len(combined_raw)}")
    print(f"   Silinən təkrar cümlələr: {duplicates_removed}")
    print(f"   Yekun təmiz cümlə sayı: {len(clean_data)}")
    
    check_label_consistency(clean_data)
    analyze_class_imbalance(clean_data, "TƏMİZLƏNMİŞ YEKUN DATASET")
    
    splits = stratified_split_for_ner(clean_data)
    
    for split_name, split_data in splits.items():
        analyze_class_imbalance(split_data, f"{split_name.upper()} SET")
        with open(f"{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
            
    print("\n✅ Bütün fayllar (train.json, val.json, test.json) uğurla yaradıldı!")

if __name__ == "__main__":
    main()
