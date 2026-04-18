# 📊 Customer Churn Prediction using PyTorch

A complete end-to-end implementation of a **Customer Churn Prediction model** built using PyTorch, covering data generation, preprocessing, model training, evaluation, and feature importance analysis.

---

## 🚀 Overview

Customer churn prediction is a critical use case in industries like telecom, SaaS, and banking, where retaining existing customers is often more cost-effective than acquiring new ones.

This project demonstrates:
- How to build a deep learning model for churn prediction
- How to train and evaluate it effectively
- How to interpret model behavior using feature importance

---

## 🧪 Dataset

The dataset is synthetically generated to simulate real-world customer behavior with **5,000 samples and 9 features**.

### Features include:
- Account Length  
- Monthly Charges  
- Total Charges  
- Number of Services  
- Contract Type  
- Payment Method  
- Tech Support  
- Online Security  

Target variable:
- **Churn (0 or 1)**

---

## ⚙️ Model Architecture

Input → 64 → 32 → 16 → 1

### Key components:
- ReLU activations
- Batch Normalization
- Dropout layers
- Sigmoid output

---

## 🏋️ Training Setup

- Loss Function: Binary Cross Entropy (BCELoss)
- Optimizer: Adam with weight decay
- Scheduler: OneCycleLR

---

## 📈 Model Performance

- Accuracy: 0.8213  
- Precision: 0.7186  
- Recall: 0.8061  
- F1 Score: 0.7599  
- ROC-AUC: 0.9024  

---

## 🔍 Feature Importance

Top features influencing churn:
1. Contract Type  
2. Monthly Charges  
3. Number of Services  

---

## ▶️ How to Run

```bash
python churn_prediction_model.py
```

Install dependencies:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

---

## 💭 Final Thought

Churn prediction is not just a modeling problem — it’s a business problem. The real value lies in understanding why customers churn and taking action.

