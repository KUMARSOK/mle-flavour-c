import requests
import time

url = "http://127.0.0.1:8000/predict"

print("--- Simulating Normal Traffic ---")
normal_reviews = [
    "The food was hot and delicious.",
    "Terrible wait times, I hated it.",
    "Friendly staff and great atmosphere."
]

for text in normal_reviews:
    res = requests.post(url, json={"text": text})
    print(f"Normal: {res.json()}")
    time.sleep(1)

print("\n--- Simulating Concept Drift (New Slang / Off-Topic) ---")
drifted_reviews = [
    "This burger is mid, no cap. Ratio.", 
    "My router keeps dropping packets on port 80.", 
    "The UI UX on this app is absolutely goated."
]

for text in drifted_reviews:
    res = requests.post(url, json={"text": text})
    print(f"Drifted: {res.json()}")
    time.sleep(1)

print("\nSimulation complete! Check data/processed/prediction_logs.csv")