import re
import string
import nltk
from nltk.corpus import stopwords

# Only download if not already present
try:
    stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text
