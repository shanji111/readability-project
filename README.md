
# 英语阅读文本可读性分析项目
成员：孙钰涵、唐莉、朱琼月
##  项目背景

本项目基于 [OneStopEnglish 分级阅读语料库](https://paperswithcode.com) 开发，旨在实现对英文阅读文本可读性难度的自动分析。  
该语料库包含 **189 篇文章的三个难度版本（初级、中级、高级）**，共计 567 篇文本，被广泛应用于可读性评估研究中。

我们使用轻量级的 DistilBERT 模型进行微调分类，实现一键预测文本难度，帮助教师快速判断教材或练习内容的适配年级。

---

## 功能介绍

- **可读性预测**：输入任意英文文本，系统返回预测的难度等级（初级 / 中级 / 高级）。
- **传统指标计算**（Flesch Score, FK Grade Level, CLI Index）
- **难词识别与释义**（基于词表与 WordNet）
---
### 💡 输出示例

输入示例：

```text
This is our English reading text readability analysis project.
```

输出结果：
```text
预测等级：Intermediate 

平均句长：9.0 词
平均词长：5.89 字母
Flesch 易读性得分：19.1（0–100，越高越容易）
FK 年级等级：12.83（约为 美国高三/预科）
Coleman–Liau 指数：15.54（估计 大学及以上）
```
##  环境依赖

### Python 版本要求：

- Python ≥ 3.8

### 安装依赖：

```bash
pip install streamlit transformers torch nltk pandas beautifulsoup4 requests
````

### 下载 NLTK 所需资源（首次运行）：

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

---

## 使用方式

### 1️⃣ 准备数据

将 OneStopEnglish 语料放入 `data/` 目录下，支持两种格式：

* 按文件夹划分：

```
data/
├── Elementary/
├── Intermediate/
└── Advanced/
```

* 或使用 `all_data.csv`（包含文本内容与对应难度标签）

确保所有文本均为 UTF-8 编码。

---

### 2️⃣ 训练模型（如需重新训练）

如需重新训练模型，运行以下命令：

```bash
python model.py
```

训练结束后，模型与分词器会保存在 `model/` 目录。若已有训练好的模型，可跳过此步骤。

---

### 3️⃣ 启动 Web 应用

```bash
streamlit run app.py
```

浏览器将自动打开本地地址（如 `http://localhost:8501`），即可使用界面进行文本分析。


---

##  输出说明

系统输出包含两个部分：

### 1. 模型预测等级

* `"Elementary"` → 初级水平
* `"Intermediate"` → 中级水平
* `"Advanced"` → 高级水平

### 2. 可读性指标说明（均为标准可解释指标）：

| 指标                 | 说明                  | 值域    | 含义                     |
| ------------------ | ------------------- | ----- | ---------------------- |
| `avg_sentence_len` | 平均句长（词数）            | -     | 数值越大说明句子结构复杂           |
| `avg_word_len`     | 平均词长（字母数）           | -     | 长词比例高时说明词汇偏专业          |
| `flesch_score`     | Flesch 阅读容易度        | 0–100 | 越高越容易读（60+ = 普通读者轻松理解） |
| `fk_grade`         | Flesch-Kincaid 年级等级 | 1–16  | 越低越容易读；12=美国高三，14=大学   |
| `cli_index`        | Coleman–Liau 指数     | 1–16  | 另一套年级估计公式，兼顾字母和句子数     |

我们为 `fk_grade` 和 `cli_index` 加入了文字提示，例如“约为 美国高三/预科”，帮助教师和学生更快做出判断。

---

## 教育适用性说明

* 该项目充分结合教育语料与机器学习方法；
* 输出指标清晰、解释丰富，适合非技术用户使用；
* 在评估阶段，我作为英语专业学生，邀请多位英语学习者和教师试用系统并提供反馈，依据其意见对界面交互、等级标签、术语翻译等进行了多轮优化。

---

##  项目结构说明

```
readability_project/
├── .gradio/ # Gradio生成的临时配置，可忽略
├── data/ # 原始数据文件夹
│ └── all_data.csv # 项目训练使用的英语文本数据集
├── model/ # 微调后模型文件
├── readability/ # 项目核心功能模块
│ ├── init.py # 包初始化
│ ├── dataset.py # 数据预处理与DataLoader构建
│ ├── predict.py # 模型加载与预测逻辑
│ ├── train.py # 模型训练主函数
│ └── utils.py # 工具函数，如可读性指标、难词提取等
├── scripts/ # 启动脚本与调试工具
│ ├── main.py # 命令行入口（输入文本后输出结果）
│ ├── web_app.py # Gradio 网页端入口（带GUI界面）
│ ├── app.py # Web UI封装组件（可选模块）
│ ├── infer.py # 单文本推理脚本（调试用）
│ ├── model.py # 模型结构定义（可选/备用）
│ └── train_model.py # 模型训练脚本（封装更完善）
└── README.md # 项目说明文档
```

---

## 📬 联系我们

如需合作或反馈建议，请联系开发者：[英语专业数据智能实践者](819916153@qq.com)

---

> “教学相长，算法向善。”
> —— 本项目致力于用人工智能辅助教学与语言学习，欢迎广大师生试用与共建 💙


