from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import csv
from datetime import datetime

app = FastAPI(title="Yelp Sentiment API", version="1.0")

MODEL_PATH = "models/logistic_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LOG_FILE = "data/processed/prediction_logs.csv"

# FIX: Force the creation of the directory before creating the file!
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Create log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "text", "prediction", "confidence"])

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    raise RuntimeError("Model artifacts not found.")

class ReviewInput(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "API is running!"}

@app.post("/predict")
def predict_sentiment(review: ReviewInput):
    if not review.text or not review.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    try:
        text_vectorized = vectorizer.transform([review.text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0].max()
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        # Log the prediction
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), review.text, sentiment, round(float(probability), 4)])
        
        return {
            "text": review.text,
            "sentiment": sentiment,
            "confidence": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))