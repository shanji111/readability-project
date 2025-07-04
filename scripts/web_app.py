import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from readability.predict import load, predict
from readability.utils import get_difficult_words, get_definition, compute_readability_report



# 可读性等级解释
LEVEL_DESCRIPTIONS = {
    "0": "初级（适合美国1-3年级）",
    "1": "中级（适合美国4-6年级）",
    "2": "高级（适合美国7-9年级及以上）"
}

# 加载模型
model, tokenizer, id2label = load("model")

def analyze(text):
    if not text.strip():
        return "请输入有效英文文本", [], {}, ""

    label_id = predict(text, model, tokenizer)
    level = id2label[str(label_id)]

    difficult_words = get_difficult_words(text)
    definitions = {w: get_definition(w) for w in difficult_words}
    metrics = compute_readability_report(text)


    # 提取字段
    avg_sent = metrics.get("avg_sentence_len", 0)
    avg_word = metrics.get("avg_word_len", 0)
    flesch = metrics.get("flesch_score", 0)
    fk = metrics.get("fk_grade", 0)
    cli = metrics.get("cli_index", 0)

    # 年级解释函数
    def explain_grade(score):
        if score < 1:
            return "幼儿园"
        elif score < 6:
            return f"小学{int(score)}年级"
        elif score < 9:
            return f"初中{int(score) - 5}年级"
        elif score < 12:
            return f"高中{int(score) - 8}年级"
        elif score < 14:
            return f"美国高三/预科"
        else:
            return "大学及以上"

    # 构造说明文本
    summary = (
        f"预测等级：{level} \n\n"
        f"平均句长：{avg_sent} 词\n"
        f"平均词长：{avg_word} 字母\n"
        f"Flesch 易读性得分：{flesch}（0–100，越高越容易）\n"
        f"FK 年级等级：{fk}（约为 {explain_grade(fk)}）\n"
        f"Coleman–Liau 指数：{cli}（估计 {explain_grade(cli)}）"
    )

    return level, difficult_words, definitions, summary


# Gradio界面构建
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="输入英文文本", lines=6, placeholder="Please enter English text here..."),
    outputs=[
        gr.Text(label="预测可读性等级"),
        gr.Textbox(label="难词列表"),
        gr.JSON(label="词义解释"),
        gr.Text(label="可读性指标报告"),
    ],
    title="📘 Readability 分析工具",
    description="输入英文段落，自动返回可读性等级（初/中/高）、难词及释义、Flesch等指标，帮助评估教材难度。"
)

if __name__ == "__main__":
    demo.launch(share=True)
