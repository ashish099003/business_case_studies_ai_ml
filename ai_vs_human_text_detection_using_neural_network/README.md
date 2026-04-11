# 🧠 AI vs Human Text Detection using Deep Learning

## 📌 Overview

This project builds a deep learning model to classify whether a given text is **AI-generated or human-written**.  
It demonstrates practical application of modern NLP techniques using TensorFlow/Keras along with key regularization and optimization strategies.

---

## 🎯 Business Problem

With the rise of generative AI systems, distinguishing between **AI-generated and human-written content** has become critical for:

- Content authenticity verification
- Academic integrity (plagiarism detection)
- Fake content / misinformation control
- Moderation systems in social platforms

---

## 🏗️ Solution Approach

We use a **Neural Network-based NLP pipeline**:

1. Text preprocessing & tokenization
2. Sequence padding
3. Embedding representation
4. Model training with regularization techniques
5. Evaluation and prediction

---

## 📊 Dataset

- Input: `text_content`
- Label:
  - `0 → Human`
  - `1 → AI`

---

## ⚙️ Model Architecture

### Basic Model
- Embedding Layer
- Flatten Layer
- Dense Layers

### Advanced Models
- GlobalAveragePooling
- Dropout Regularization
- Batch Normalization
- Early Stopping

---

# 🔑 Key Concepts Explained (Why, What, When, How)

---

## 1️⃣ Embedding Layer

### What
Transforms words into **dense vector representations**.

### Why
- Raw text cannot be used directly
- One-hot encoding is inefficient
- Captures semantic meaning

### When
- Any NLP task

### How
Each word index → mapped to trainable vector

---

## 2️⃣ Dropout

### What
Randomly disables neurons during training.

### Why
- Prevents overfitting

### When
- When validation performance is worse than training

### How
Dropout(0.5) → 50% neurons dropped

---

## 3️⃣ Batch Normalization

### What
Normalizes layer inputs

### Why
- Speeds up training
- Stabilizes learning

### When
- Deep neural networks

### How
Normalize → Scale → Shift

---

## 4️⃣ L2 Regularization

### What
Penalty for large weights

### Why
- Prevents overfitting

### When
- Complex models

### How
Loss = Loss + λ * weights²

---

## 5️⃣ Early Stopping

### What
Stops training early

### Why
- Prevents overfitting

### When
- Validation loss increases

### How
Monitor validation performance

---

## 🚀 How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## 📌 Author

Ashish Sinha
