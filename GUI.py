import tkinter as tk
from tkinter import messagebox
import joblib
from utils.preprocessing import clean_text
import csv
import os
from datetime import datetime
import webbrowser

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Predict sentiment and save to CSV
def predict_sentiment():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter some text!")
        return

    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = round(max(probabilities) * 100, 2)

    if prediction == 1:
        result_text = f"😊 Sentiment: Positive ({confidence}%)"
        sentiment_label = "Positive"
        color = "green"
    else:
        result_text = f"☹️ Sentiment: Negative ({confidence}%)"
        sentiment_label = "Negative"
        color = "red"

    result_label.config(text=result_text, fg=color)

    # Save prediction to CSV
    save_path = "predictions.csv"
    file_exists = os.path.isfile(save_path)
    with open(save_path, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Input Text", "Predicted Sentiment", "Confidence (%)"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, sentiment_label, confidence])

# Open predictions.csv
def open_history():
    file_path = os.path.abspath("predictions.csv")
    if os.path.exists(file_path):
        webbrowser.open(f"file://{file_path}")
    else:
        messagebox.showinfo("No History", "No prediction history found yet.")

# Setup GUI
root = tk.Tk()
root.title("Sentiment Analyzer")
root.geometry("420x350")
root.configure(bg="#f2f2f2")

tk.Label(root, text="Enter your sentence below:", bg="#f2f2f2", font=("Arial", 12)).pack(pady=10)

entry = tk.Text(root, height=5, width=45, font=("Arial", 11))
entry.pack(pady=10)

predict_button = tk.Button(root, text="Analyze Sentiment", command=predict_sentiment, font=("Arial", 11), bg="#007acc", fg="white")
predict_button.pack(pady=5)

view_button = tk.Button(root, text="📄 View Prediction History", command=open_history, font=("Arial", 11), bg="#555555", fg="white")
view_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f2f2f2")
result_label.pack(pady=20)

root.mainloop()
