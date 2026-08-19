import os
import re
import pandas as pd
import pickle
import urllib.request
import zipfile
import shutil
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
FINAL_CSV_PATH = "data/raw/yelp_reviews.csv"

def clean_text(text):
    """Removes punctuation and converts to lowercase."""
    text = re.sub(r'[^\w\s]', '', str(text))
    return text.lower()

def main():
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("1. Ingesting Yelp Data from official UCI Academic Repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
    zip_path = "data/raw/temp.zip"
    
    # Download and extract
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")
        
    # Read the extracted Yelp file
    extracted_file = "data/raw/sentiment labelled sentences/yelp_labelled.txt"
    df = pd.read_csv(extracted_file, sep='\t', header=None, names=['text', 'label'])
    
    # Clean up the zip and extracted folder to keep the workspace neat
    os.remove(zip_path)
    shutil.rmtree("data/raw/sentiment labelled sentences")
    if os.path.exists("data/raw/__MACOSX"):
        shutil.rmtree("data/raw/__MACOSX")

    # Save exactly where DVC expects it
    df.to_csv(FINAL_CSV_PATH, index=False)
    print(f"Raw Yelp data saved to {FINAL_CSV_PATH}")

    print("2. Cleaning Data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    print("3. Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )

    print("4. Feature Engineering (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("5. Saving Processed Data and Vectorizer...")
    # Save processed data
    train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    train_df['label'] = y_train.values
    test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    test_df['label'] = y_test.values

    train_df.to_csv(f"{PROCESSED_DATA_DIR}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DATA_DIR}/test.csv", index=False)

    # Save the vectorizer
    with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Phase 1 Complete with Yelp Data!")

if __name__ == "__main__":
    main()