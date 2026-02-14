# 🔐 IoT Network Intrusion Detection using MLP + RAG Explainability

---

## 📌 1. Project Overview

This project presents a complete **IoT Network Traffic Classification System** built using a **Multi-Layer Perceptron (MLP)** trained on the Stratosphere Laboratory **IoT-23 dataset**.

The system performs **multi-class classification of IoT network flows** and integrates a **Retrieval-Augmented Generation (RAG)** layer to provide human-readable explanations for each prediction.

### 🎯 Objective

* Detect malicious IoT traffic
* Classify attack types accurately
* Provide interpretable AI-driven explanations
* Maintain real-time inference capability

---

## 🧠 2. Core Detection Engine — Multi-Layer Perceptron (MLP)

The primary classification model is a deep neural network implemented using **TensorFlow/Keras**.

### 🔹 Network Architecture

```
Input Layer (64 Features)
        ↓
Dense (512) + ReLU
        ↓
Dropout (0.3–0.5) + L2 Regularization
        ↓
Dense (256) + ReLU
        ↓
Dropout + Batch Normalization
        ↓
Dense (128) + ReLU
        ↓
Dropout
        ↓
Output Layer (Softmax – Multi-Class Classification)
```

### 🔹 Model Configuration

* Loss Function: Categorical Cross-Entropy
* Optimizer: Adam (Learning Rate = 0.001)
* L2 Regularization: λ = 0.001
* Dropout: 0.3 – 0.5
* Early Stopping: Patience = 15 epochs
* Batch Normalization for training stability

### ⚙ Design Rationale

The architecture is designed to:

* Capture non-linear traffic behavior
* Prevent overfitting via dropout + L2
* Maintain computational efficiency
* Enable real-time classification

---

## 📊 3. Dataset

**Dataset Used:** IoT-23
Source: Stratosphere Laboratory

### 📁 Dataset Characteristics

* 20 malicious capture scenarios
* 3 benign capture scenarios
* ~2 million labeled traffic flows
* 12 core network flow features

### 🛡 Attack Categories

* Benign
* DDoS
* Command & Control (C&C)
* Port Scan
* Mirai
* Okiru
* Torii
* File Download
* Heartbeat

The dataset reflects real-world IoT malware behaviors including botnets and distributed attacks.

---

## 🔄 4. Data Preprocessing Pipeline

The preprocessing pipeline ensures clean, normalized, and ML-ready input:

1. Merged multiple CSV captures
2. Normalized inconsistent label names
3. Removed irrelevant columns (ts, uid, service, etc.)
4. Hashed IP addresses for anonymization
5. Encoded categorical features (proto, conn_state, history)
6. Converted numeric columns to proper format
7. Standardized features using `StandardScaler`
8. Custom dataset split:

   * 60% Training
   * 20% Validation
   * 20% Testing

---

## 📈 5. Model Performance & Evaluation

The MLP demonstrated:

* Stable convergence during training
* Decreasing training & validation loss
* Balanced precision across attack classes
* Strong generalization on unseen test data

### 📌 Evaluation Metrics

* Test Loss
* Accuracy
* Precision

Regularization and early stopping effectively minimized overfitting while preserving high detection capability.

---

## 🔍 6. Explainability Layer — RAG Integration

To address the black-box nature of neural networks, a **Retrieval-Augmented Generation (RAG)** module was integrated.

### 🔹 Workflow

1. MLP predicts traffic label
2. Label is mapped to a cybersecurity knowledge base
3. Contextual information is retrieved
4. Google Gemini API generates a human-readable explanation

### 💬 Example Output

> “The traffic is classified as DDoS due to unusually high packet frequency directed toward a single IP address with abnormal response byte distribution, which indicates a distributed denial-of-service attack pattern.”

### 🎯 Benefit

This integration bridges:

* High-performance deep learning
* Analyst-friendly interpretability
* Transparent decision-making

---

## 🛠 7. Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas / NumPy
* Matplotlib / Seaborn
* Google Colab
* Google Gemini API

---

## 🎯 8. Key Contributions

✔ Multi-class IoT intrusion detection using MLP
✔ Real-world dataset integration (IoT-23)
✔ Full ML lifecycle implementation
✔ Regularized deep neural architecture
✔ RAG-based explainable AI layer
✔ Scalable and deployment-ready design

---

## 🚀 9. Future Work

* Replace MLP with Transformer-based architectures
* Integrate SHAP/LIME for feature-level explainability
* Real-time streaming traffic classification
* Deploy via REST API or web dashboard
* Benchmark against CNN/Hybrid IDS models

---


Tell me and I’ll tailor it.
