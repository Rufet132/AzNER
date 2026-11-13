import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_PATH = "./final_az_ner_model"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

ner_pipeline = pipeline(
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

def extract_entities(text):
    if not text.strip():
        return {"text": "", "entities": []}, []
    
    raw_results = ner_pipeline(text)
    clean_results = clean_and_merge_entities(text, raw_results)
    
    entities = []
    table_rows = []
    
    for item in clean_results:
        clean_word = item["word"].rstrip(",.!?;:")
        entities.append({
            "entity": item["entity_group"],
            "start": item["start"],
            "end": item["start"] + len(clean_word)
        })
        table_rows.append([
            clean_word,
            item["entity_group"],
            f"{item['score']:.2%}"
        ])
        
    return {"text": text, "entities": entities}, table_rows

color_map = {
    "PER": "#3b82f6",      
    "LOC": "#10b981",      
    "ORG": "#f59e0b",      
    "DATE": "#8b5cf6",     
    "PRODUCT": "#ef4444"   
}

examples = [
    ["Sənan Hacıyev, Azərbaycanın Səngəçal bölgəsində, Qazi Marketdən Asus Tuf F15 aldı."],
    ["Apple şirkəti 2026-cı ilin sentyabrında Bakıda yeni iPhone 18 Pro modelini təqdim etdi."],
    ["SOCAR və BP arasında Xəzər dənizində yeni layihə üzrə müqavilə imzalanıb."]
]

with gr.Blocks(title="AzNER - Azerbaijani Named Entity Recognition") as demo:
    gr.Markdown("# 🇦🇿 AzNER: Azerbaijani Named Entity Recognition")
    gr.Markdown("Xəbər mətnini daxil edin və modelin **PERSON, LOCATION, ORGANIZATION, DATE, PRODUCT** kateqoriyalarını necə tanıdığını yoxlayın.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=5,
                placeholder="Azərbaycan dilində mətn daxil edin...",
                label="Daxil edilən Mətn"
            )
            submit_btn = gr.Button("Analiz Et (Extract Entities)", variant="primary")
            gr.Examples(examples=examples, inputs=input_text)
            
        with gr.Column():
            highlighted_output = gr.HighlightedText(
                label="Nəticə (Vizual)",
                combine_adjacent=True,
                show_legend=True,
                color_map=color_map
            )
            table_output = gr.Dataframe(
                headers=["Entity", "Kateqoriya", "Eminlik (Confidence)"],
                label="Detallı Cədvəl"
            )
            
    submit_btn.click(
        fn=extract_entities,
        inputs=input_text,
        outputs=[highlighted_output, table_output]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
