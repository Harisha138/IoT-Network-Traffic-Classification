🔐 IoT Network Intrusion Detection using MLP + RAG Explainability
📌 Project Overview

This project presents an end-to-end IoT Network Traffic Classification System using a Multi-Layer Perceptron (MLP) neural network trained on the IoT-23 dataset.

The system classifies IoT network flows into multiple categories such as:

Benign

DDoS

Command & Control (C&C)

Port Scan

Mirai

Okiru

Torii

File Download

Heartbeat

To improve interpretability, the model is integrated with a Retrieval-Augmented Generation (RAG) layer that generates human-readable explanations for predictions.

🧠 Core Focus: Multi-Layer Perceptron (MLP)

The primary detection engine is a deep neural network built using TensorFlow/Keras.

🔹 Architecture
Input (64 features)
   ↓
Dense (512) + ReLU + Dropout
   ↓
Dense (256) + ReLU + Dropout
   ↓
Dense (128) + ReLU + Dropout
   ↓
Output Layer (Softmax – Multi-class classification)

🔹 Regularization & Optimization

L2 Regularization (λ = 0.001)

Dropout (0.3 – 0.5)

Batch Normalization

Adam Optimizer (LR = 0.001)

Early Stopping (patience = 15)

This architecture balances:

High accuracy

Generalization

Computational efficiency

Real-time inference capability

📊 Dataset

Dataset Used: IoT-23 (Stratosphere Laboratory)

20 malicious captures

3 benign captures

2M+ traffic records

12 key network flow features

The dataset contains real IoT malware scenarios including:

Mirai botnet

Okiru botnet

DDoS attacks

C&C communications

Port scanning

🔄 Data Preprocessing Pipeline

Merged multiple CSV files

Cleaned inconsistent label formats

Removed irrelevant columns

Hashed IP addresses for privacy

Encoded categorical variables

Standardized numerical features using StandardScaler

Split dataset (60% train, 20% validation, 20% test)

📈 Model Performance

The MLP demonstrated:

Strong convergence during training

Stable validation performance

Effective generalization on unseen test data

Balanced accuracy and precision across classes

Regularization and dropout reduced overfitting while maintaining strong classification capability.

🔍 Explainability Layer (RAG Integration)

To address the black-box nature of neural networks:

The predicted label is passed to a knowledge base

Relevant cybersecurity information is retrieved

Google Gemini API generates a natural-language explanation

Example Output:

“The traffic is classified as DDoS because of high packet volume directed toward a single IP address with abnormal response byte distribution, indicating a distributed denial-of-service behavior.”

This bridges the gap between:

High-performance deep learning

Human-interpretable security analysis

🛠 Tech Stack

Python

TensorFlow / Keras

Scikit-learn

Pandas / NumPy

Google Colab

Google Gemini API (RAG)

Matplotlib / Seaborn

🎯 Key Contributions

✔ Multi-class IoT malware detection using MLP
✔ End-to-end ML pipeline
✔ Real-world dataset usage (IoT-23)
✔ Regularized deep neural architecture
✔ Explainable AI integration using RAG
✔ Deployment-ready model artifacts

🚀 Future Work

Replace MLP with Transformer-based architecture

Real-time streaming intrusion detection

Deploy via REST API or dashboard

Integrate SHAP/LIME for feature-level explanations
