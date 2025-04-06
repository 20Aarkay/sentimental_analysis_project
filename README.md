# 💬 Sentiment Analysis using Machine Learning & Tkinter GUI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

A complete sentiment analysis project that uses machine learning (Logistic Regression) to classify Twitter data as **positive** or **negative**. It includes a training pipeline, evaluation script, and an easy-to-use **Tkinter GUI** for real-time predictions.

---

## 🖼️ Demo

<img src="https://user-images.githubusercontent.com/your-screenshot-url/demo.gif" width="500">

---

## 📁 Project Structure

```
sentimental-analysis-project/
│
├── data/                        # All CSV datasets
│   ├── training.1600000.processed.noemoticon.csv
│   ├── train.csv
│   ├── test.csv
│   └── testdata.manual.2009.06.14.csv
│
├── models/                      # Saved ML model
│   └── sentiment_model.pkl
│
├── utils/
│   └── preprocessing.py         # Text cleaning & preprocessing
│
├── main.py                      # Train and save model
├── evaluate.py                  # Evaluate model on test set
├── gui.py                       # Tkinter GUI for sentiment prediction
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Features

- ✅ Trains on 1.6M+ tweets from Sentiment140
- ✅ Cleans tweets using NLTK and regex
- ✅ TF-IDF vectorization of tweets
- ✅ Logistic Regression classifier
- ✅ Model evaluation with metrics
- ✅ GUI for user-friendly predictions
- ✅ Modular and extendable codebase

---

## 🔧 Installation

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/sentimental-analysis-project.git
cd sentimental-analysis-project
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Download NLTK resources (only once)**  
```python
import nltk
nltk.download('stopwords')
```

---

## 📊 Usage

### 1. 🔁 Train the Model
```bash
python main.py
```
This will train the sentiment model and save it as `models/sentiment_model.pkl`.

---

### 2. 🧪 Evaluate on Test Data
```bash
python evaluate.py
```
Evaluates model on manual test dataset and prints metrics.

---

### 3. 🖥️ Launch the GUI
```bash
python gui.py
```
Enter any tweet/text, click **Predict Sentiment**, and get instant results.

---

## 🧹 Preprocessing

Text is cleaned by:
- Lowercasing
- Removing links, usernames, hashtags
- Removing punctuation
- Filtering out stopwords

---

## 📦 Requirements

```
scikit-learn
pandas
nltk
joblib
tkinter (comes preinstalled with Python)
```

You can install them with:

```bash
pip install -r requirements.txt
```

---

## ✨ Sample Predictions

| Tweet                                | Prediction |
|--------------------------------------|------------|
| I love the new update!               | 👍 Positive |
| Worst experience ever. I'm done.     | 👎 Negative |
| The movie was okay, not the best.    | 👎 Negative |
| Feeling super excited today!         | 👍 Positive |

---

## 🚀 Future Ideas

- Add Flask/Streamlit web interface
- Use LSTM or BERT for better accuracy
- Extend to multilingual sentiment analysis
- Live tweet prediction using Twitter API

---

## 🙋‍♂️ Author

**Ram Krishna Singh**  
📫 [ramkrishnasingh094@gmail.com](mailto:ramkrishnasingh094@gmail.com)  
🌍 India

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ⭐️ Show Your Support

If you found this project helpful, please give it a ⭐️ on GitHub and share it!
