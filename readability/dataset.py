# readability/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

class ReadabilityDataset(Dataset):
    """
    将 CSV 数据包装成 HuggingFace Trainer 可用的 Dataset。
    自动生成 label2id / id2label。
    """
    def __init__(self, csv_path, text_col="content", label_col="level",
                 tokenizer=None, max_len=256):
        df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
        self.texts = df[text_col].astype(str).tolist()
        self.labels_str = df[label_col].astype(str).tolist()

        # 标签映射
        unique = sorted(set(l.title() for l in self.labels_str))
        self.label2id = {lab: idx for idx, lab in enumerate(unique)}
        self.id2label = {idx: lab for lab, idx in self.label2id.items()}
        self.labels = [self.label2id[l.title()] for l in self.labels_str]

        self.tokenizer = tokenizer
        self.encodings = (tokenizer(self.texts,
                                    truncation=True,
                                    padding=True,
                                    max_length=max_len)
                          if tokenizer else None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.tokenizer:
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        return self.texts[idx], self.labels[idx]
