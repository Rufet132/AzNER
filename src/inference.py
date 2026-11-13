import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_PATH = "./final_az_ner_model"

def load_ner_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=device
    )

def clean_and_merge_entities(text, raw_results):
    if not raw_results:
        return []
        
    merged = []
    for item in raw_results:
        if not merged:
            merged.append(dict(item))
            continue
            
        prev = merged[-1]
        gap = text[prev["end"]:item["start"]]
        if prev["entity_group"] == item["entity_group"] and gap == "":
            prev["end"] = item["end"]
            prev["word"] = text[prev["start"]:prev["end"]]
            prev["score"] = (prev["score"] + item["score"]) / 2
        else:
            item["word"] = text[item["start"]:item["end"]]
            merged.append(dict(item))
            
    merged[0]["word"] = text[merged[0]["start"]:merged[0]["end"]]
    return merged

def predict_entities(ner_pipe, text):
    raw_results = ner_pipe(text)
    results = clean_and_merge_entities(text, raw_results)
    
    print(f"\n📝 Mətn: {text}")
    print("-" * 60)
    if not results:
        print("Heç bir entity tapılmadı.")
        return
    for ent in results:
        print(f"  {ent['word']:<25} | {ent['entity_group']:<10} | Eminlik: {ent['score']:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    ner_pipe = load_ner_pipeline()
    
    test_sentences = [
        "Sanan Haciyev, Azerbaycanin Sanqacal bolgesinde, Qazi Marketden Asus Tuf F15 aldi.",
        "Sənan Hacıyev, Azərbaycanın Səngəçal bölgəsində, Qazi Marketdən Asus Tuf F15 aldı.",
        "Sənan Hajiyev, Azerbaycanin Sanqacal bolgəsinde, Qazi Marketdən Asus Tuf F15 aldi.",
        "Aselsan şirkəti yeni Galaxy S26 Ultra smartfonunu bazara çıxarmağa hazırlaşır.",
        "Türk Hava Yolları (THY) Qahirə istiqamətində yeni reyslər açır."
    ]
    
    for txt in test_sentences:
        predict_entities(ner_pipe, txt)
