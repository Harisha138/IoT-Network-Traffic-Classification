The rapid growth in the number of Internet of Things devices has increased network complexity and cybersecurity vulnerabilities while demanding highly effective traffic classification systems for intrusion detection and anomaly identification. This paper proposes a novel hybrid approach, which integrates deep neural networks with Retrieval-Augmented Generation for the classification of IoT network traffic, achieving both high accuracy and explainability. Our proposed architecture employs a deep neural classifier with dense layers (512→256→128→output) enhanced with dropout and L2 regularization for feature learning. It is integrated with the RAG system, which retrieves contextual threat intelligence and generates natural language explanations for classification decisions. The

The proposed system architecture includes the following key stages:
1.
Data Ingestion and Cleaning
➢
Loads multiple CSV files from the IoT-23 dataset directory.
➢
Drops irrelevant columns (e.g., service, uid, ts, duration).
➢
Cleans and normalizes the label field by merging similar attack types under unified names (e.g., “Malicious DDo” → “Malicious DDoS”).
➢
Removes missing or malformed entries.
2.
Feature Engineering and Encoding
➢
Hashes IP addresses (id.orig_h, id.resp_h) for privacy.
➢
Encodes categorical columns (proto, conn_state, history) using LabelEncoder.
➢
Converts numeric fields (id.orig_p, id.resp_p, orig_ip_bytes, resp_pkts, resp_ip_bytes) to proper numeric types.
➢
Scales all numerical features using StandardScaler for normalized input.
3.
Dataset Splitting
➢
Custom train_val_test_split() function splits the data into 60% training, 20% validation, and 20% testing sets.
➢
The remove_labels() function separates feature matrices (X) and label vectors (y).
4.
Neural Network Model
➢
Built using TensorFlow Keras Sequential API:
Input → Dense(512, ReLU, Dropout 0.5)
→ Dense(256, ReLU, Dropout 0.5)
→ Dense(128, ReLU, Dropout 0.5)
→ Output(Dense, Softmax)
➢
Uses L2 regularization, Adam optimizer, and categorical cross-entropy loss.
➢
Includes EarlyStopping to avoid overfitting.
5.
Evaluation and Saving
➢
Evaluates model accuracy, error, and precision on the test dataset.
➢
Saves model, weights, label encoders, and scalers for reuse.
6.
RAG + Gemini Integration for Explainability
➢
The model’s output is passed through a knowledge base dictionary (RAG_KNOWLEDGE_BASE) describing each attack type.
➢
Google’s Gemini API generates a detailed, context-aware explanation for each prediction, highlighting which input features contributed to the classification.
➢
The interface is built using ipywidgets for interactive testing.
Layer Configuration:
•
Input Layer: Accepts 64-dimensional feature vectors derived from network flow statistics
•
Hidden Layer 1: 512 neurons with ReLU activation and 0.3 dropout rate
•
Hidden Layer 2: 256 neurons with ReLU activation and 0.4 dropout rate
•
Hidden Layer 3: 128 neurons with ReLU activation and 0.5 dropout rate
•
Output Layer: Softmax activation for multi-class probability distribution
Regularization and Optimization:
•
L2 regularization (λ = 0.001) applied to all dense layers to prevent overfitting
•
Batch normalization between hidden layers for training stability
•
Adam optimizer with learning rate scheduling (initial: 0.001, decay: 0.95)
•
Early stopping with validation loss monitoring (patience: 15 epochs)
The network architecture balances model complexity with computational efficiency, enabling real-time inference while maintaining sufficient capacity for complex pattern recognition in diverse IoT traffic scenarios.
