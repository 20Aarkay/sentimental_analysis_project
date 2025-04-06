import pandas as pd
from utils.preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
df_main = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None, low_memory=False)
df_main.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df_main = df_main[['target', 'text']]

# Drop non-numeric target values
df_main = df_main[pd.to_numeric(df_main['target'], errors='coerce').notnull()]
df_main['target'] = df_main['target'].astype(int)

# Keep only 0 (negative) and 4 (positive) sentiment
df_main = df_main[df_main['target'].isin([0, 4])]

# Convert 4 → 1 for binary classification
df_main['target'] = df_main['target'].replace({4: 1})

# Clean text
df_main['clean_text'] = df_main['text'].apply(clean_text)

# Drop rows with missing values
df_main.dropna(subset=['clean_text', 'target'], inplace=True)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_main['clean_text'])
y = df_main['target']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_val)
print("Classification Report on Validation Set:")
print(classification_report(y_val, y_pred))

import os

# Create folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model and vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")


print("✅ Model training complete. Artifacts saved to 'models/'")
