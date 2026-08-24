# Model Comparison Report & Deployment Justification

## Experiments Conducted
During Week 2, two models were trained and tracked using MLflow on the Yelp polarity dataset:
1. **Classical ML (Logistic Regression):** Trained using TF-IDF features.
2. **Transformer (DistilBERT):** Fine-tuned using Hugging Face and a native PyTorch dataset.

## Results
* **Logistic Regression:** Accuracy: ~78.5% | F1-Score: ~77.0% 
* **DistilBERT:** Accuracy: ~86.0% | F1-Score: ~86.0% 

## Justification for Deployment (Best Model)
Although the fine-tuned DistilBERT model achieved a higher accuracy (86% vs 78.5%), the **Logistic Regression model was selected as the best model for production deployment in Phase 3**. 

**Reasoning:**
1. **Latency & Throughput:** The API requires real-time text classification. Logistic Regression provides sub-millisecond inference times, whereas the Transformer introduces significant latency, especially on CPU-only infrastructure.
2. **Resource Constraints (Docker):** The Logistic Regression model and its TF-IDF vectorizer are just a few kilobytes. The DistilBERT model requires >250MB of disk space and heavy dependencies (PyTorch), which would bloat the Docker container and increase hosting costs.
3. **Complexity vs. Value:** For a simple positive/negative sentiment task, a 7.5% increase in accuracy does not justify the massive increase in architectural complexity, memory footprint, and compute requirements.