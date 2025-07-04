# readability/train.py
import os, json, torch
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          TrainingArguments, Trainer)
from torch.utils.data import random_split
from .dataset import ReadabilityDataset

def train(csv_path="data/converted_all_data.csv",
          output_dir="model",
          num_epochs=3,
          batch_size=8,
          max_len=256,
          seed=42,
          use_cpu=True):

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    full_ds = ReadabilityDataset(csv_path, tokenizer=tokenizer, max_len=max_len)

    # 划分 90% 训练 / 10% 验证
    torch.manual_seed(seed)
    val_size = max(1, int(0.1 * len(full_ds)))
    train_ds, val_ds = random_split(full_ds,
                                    [len(full_ds)-val_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))

    print(f"📊 训练 {len(train_ds)} 条 | 验证 {len(val_ds)} 条")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(full_ds.label2id),
        id2label=full_ds.id2label,
        label2id=full_ds.label2id
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        no_cuda=use_cpu,
        seed=seed
    )

    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_ds,
                      eval_dataset=val_ds,
                      tokenizer=tokenizer)

    print("🚀 开始训练 …")
    trainer.train()
    print("✅ 训练完成！")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "id2label.json"), "w",
              encoding="utf-8") as f:
        json.dump(full_ds.id2label, f, ensure_ascii=False, indent=2)

    print("💾 模型已保存到", output_dir)
