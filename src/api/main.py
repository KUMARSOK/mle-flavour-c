from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

app = FastAPI(title="Yelp Sentiment API", version="1.0")

MODEL_PATH = "models/logistic_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

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
        
        return {
            "text": review.text,
            "sentiment": sentiment,
            "confidence": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))