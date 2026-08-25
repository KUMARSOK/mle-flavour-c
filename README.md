"# mle-flavour-c" 

# Yelp Sentiment Analysis ML Pipeline

An end-to-end machine learning pipeline for classifying the sentiment of Yelp reviews. 

## Architecture Overview
1. **Data Engineering (DVC):** Fetches the UCI Yelp dataset, cleans text, applies TF-IDF, and tracks versions using DVC.
2. **Experimentation (MLflow):** Compares a Logistic Regression model (baseline) against a fine-tuned DistilBERT transformer. 
3. **Deployment (FastAPI & Docker):** Serves the best, low-latency model via a REST API with edge-case handling.
4. **Monitoring:** Logs all predictions and simulates concept drift.

## How to Run Locally

### 1. Install Dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 2. Run the API (Local)
uvicorn src.api.main:app --reload

Access the interactive dashboard at http://127.0.0.1:8000/docs

###3. Run with Docker
docker build -t yelp-sentiment-api .
docker run -p 8000:8000 yelp-sentiment-api




