import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")


from model import load_trained_model, predict_readability
from readability_project.readability.utils import get_difficult_words, get_definition, compute_readability_metrics

if __name__ == "__main__":
    model, tokenizer = load_trained_model("model")

    sample_text = "The process of photosynthesis in plants is essential to life on Earth."

    label = predict_readability(sample_text, model, tokenizer)

    label_map = {
        0: "高级（Advanced）",
        1: "初级（Elementary）",
        2: "中级（Intermediate）"
    }
    print("预测等级：", label_map.get(label, "未知"))

    diff_words = get_difficult_words(sample_text)
    print("难词：", diff_words)

    for word in diff_words:
        print(word, "释义：", get_definition(word))

    print("可读性指标：", compute_readability_metrics(sample_text))
