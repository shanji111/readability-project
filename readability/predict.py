import json, torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AutoConfig

def load(model_dir="model"):
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)  # 💡 明确使用本地 config
    model = DistilBertForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        use_safetensors=True
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir, local_files_only=True)

    with open(f"{model_dir}/id2label.json", "r", encoding="utf-8") as f:
        id2label = json.load(f)

    model.eval()
    return model, tokenizer, id2label



def predict(text, model, tokenizer):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        return int(torch.argmax(logits, dim=1))


