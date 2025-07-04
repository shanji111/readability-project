
import transformers, accelerate, inspect
print("transformers =", transformers.__version__)
print("accelerate   =", accelerate.__version__)
print("TrainingArguments from", inspect.getfile(transformers.TrainingArguments))

# model.py  ——  适配 data/all_data.csv 的训练 + 推理脚本
import os
import random
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset, random_split

# ===== 可调参数 =====
DATA_PATH = "../data/all_data.csv"  # 你的大 csv 文件
TEXT_COL = "content"                 # csv 里的“文本列”列名
LABEL_COL = "level"                  # csv 里的“标签列”列名
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 256                        # 句子较长可以酌情调 512

# ===================

class ReadabilityDataset(Dataset):
    """把文本+标签打包成 HuggingFace Trainer 需要的 Dataset。"""
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(texts,
                                   truncation=True,
                                   padding=True,
                                   max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_csv_dataset(path: str, text_col: str, label_col: str):
    """读取 csv，返回 texts, labels (标签仍是字符串)"""
    df = pd.read_csv(path).dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels


def train_and_save():
    # ---------- 1. 读数据 ----------
    print("📥 读取数据中…")
    texts, labels_str = load_csv_dataset(DATA_PATH, TEXT_COL, LABEL_COL)

    # ---------- 2. 构建标签映射 ----------
    unique_labels = sorted(set(lab.lower().capitalize() for lab in labels_str))
    label2id = {lab: idx for idx, lab in enumerate(unique_labels)}
    id2label = {idx: lab for lab, idx in label2id.items()}
    numeric_labels = [label2id[lab.lower().capitalize()] for lab in labels_str]

    print(f"🔖 标签映射: {label2id}")

    # ---------- 3. 划分训练/验证 ----------
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset_full = ReadabilityDataset(texts, numeric_labels, tokenizer)

    val_size = max(1, int(0.1 * len(dataset_full)))  # 至少留 1 条
    train_size = len(dataset_full) - val_size
    train_dataset, val_dataset = random_split(
        dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"📊 训练集 {len(train_dataset)} 条, 验证集 {len(val_dataset)} 条")

    # ---------- 4. 初始化模型 ----------
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # ---------- 5. Trainer ----------
    training_args = TrainingArguments(
        output_dir="../model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_dir="logs",
        logging_steps=20,
        save_total_limit=1,
        no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset  # ✅ 否则 metric_for_best_model 无法使用
    )

    # ---------- 6. 开始训练 ----------
    print("🚀 开始训练…")
    trainer.train()
    print("✅ 训练完成！")

    # ---------- 7. 保存模型 ----------
    print("💾 正在保存模型到 model/ …")
    trainer.save_model("model")
    tokenizer.save_pretrained("model")
    print("🎉 模型已保存，后续可 load_trained_model 使用！")


# ===== 推理用函数（和以前一致） =====
def load_trained_model(model_path="model"):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def predict_readability(text: str, model, tokenizer):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return int(torch.argmax(logits, dim=1))


# ===== 主入口 =====
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到 {DATA_PATH}，请确认 csv 已放入 data/ 目录")
    train_and_save()
