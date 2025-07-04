import sys
from readability.predict import load, predict

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python infer.py \"英文文本\"")
        sys.exit(0)

    text = sys.argv[1]
    model, tokenizer, id2label = load("readability_project/model")
    label_id = predict(text, model, tokenizer)
    print("预测等级：", id2label.get(str(label_id), label_id))
