# Model Monitoring and Retraining Strategy

## 1. Monitoring Metrics
The deployed API logs every incoming request, including the timestamp, input text, predicted sentiment, and model confidence score to `prediction_logs.csv`.

## 2. Drift Detection
We simulate concept drift by injecting modern slang (e.g., "mid", "goated") and out-of-domain text (e.g., tech support tickets). Because the model was trained on standard restaurant reviews, its confidence scores on drifted text will behave unpredictably, indicating data drift.

## 3. Retraining Trigger Design
To maintain model health, we will implement the following automated triggers:
* **Statistical Trigger:** If the average daily prediction confidence drops below **65%** over a rolling 48-hour window, an alert is sent to the engineering team.
* **Data Volume Trigger:** Once 5,000 new reviews are collected and manually labeled, a DVC pipeline will automatically trigger a new MLflow training run.
* **Shadow Deployment:** The newly trained model will run in "shadow mode" (making predictions without serving them to users). If the new model beats the current production model's F1-score by at least 2%, it will be promoted to production via Docker registry swap.