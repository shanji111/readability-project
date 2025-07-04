import sys
import os

# 添加 readability_project 到 sys.path（以便识别 readability）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from readability.predict import load, predict
from readability.utils import get_difficult_words, get_definition, compute_readability_metrics

def main():
    model, tokenizer, id2label = load("readability_project/model")
    text = input("请输入英语文本：\n")
    label_id = predict(text, model, tokenizer)
    print("预测等级：", id2label[str(label_id)])

    diff_words = get_difficult_words(text)
    print("难词：", diff_words)
    for word in diff_words:
        print(f"{word} 的释义：{get_definition(word)}")

    print("可读性指标：", compute_readability_metrics(text))

if __name__ == "__main__":
    main()
