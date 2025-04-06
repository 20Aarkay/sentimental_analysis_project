import pandas as pd
import joblib
from utils.preprocessing import clean_text

# Load saved model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load manual test dataset
test_df = pd.read_csv("data/testdata.manual.2009.06.14.csv", encoding='latin-1', header=None)
test_df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
test_df = test_df[['target', 'text']]

# Convert 4 → 1, drop invalid labels
test_df = test_df[test_df['target'].isin([0, 4])]
test_df['target'] = test_df['target'].replace({4: 1})
test_df['clean_text'] = test_df['text'].apply(clean_text)

# Drop missing values
test_df.dropna(subset=['clean_text'], inplace=True)

# Vectorize using the same vectorizer
X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['target']

# Predict and evaluate
y_pred_test = model.predict(X_test)

from sklearn.metrics import classification_report
print("📊 Evaluation on Manual Test Dataset:")
print(classification_report(y_test, y_pred_test))
