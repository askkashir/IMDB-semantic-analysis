# Required installations (only once):
# pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn
# python -m spacy download en_core_web_sm

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load spaCy English model
import en_core_web_sm
nlp = en_core_web_sm.load()

# Load dataset
df = pd.read_csv('dataset.csv')  # Your file must have 'review' and 'sentiment' columns

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):
        return ""

    text = re.sub(r'<.*?>', '', text)               # Remove HTML tags
    text = text.lower()                             # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)            # Remove numbers & punctuation
    tokens = word_tokenize(text)                    # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    doc = nlp(' '.join(tokens))                     # Lemmatize
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

# Drop null rows and sample
df = df.dropna(subset=['review', 'sentiment'])
df = df.sample(n=min(500, len(df))).copy()  # Limit to 500 rows

# Clean text
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Encode sentiment
df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'negative': 0})
X = df['cleaned_review']
y = df['sentiment_numeric']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_tfidf, y_train)

# Predict function
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Sample predictions
print(predict_sentiment("This movie was absolutely fantastic!"))
print(predict_sentiment("I was so bored throughout the entire film."))
print(predict_sentiment("The film was okay, not great but not terrible."))

# Evaluation
y_pred = model.predict(X_test_tfidf)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
