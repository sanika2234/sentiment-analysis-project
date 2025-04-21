import numpy as np
import pandas as pd
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')

# Define global stopwords and punctuation
stop_words = set(stopwords.words("english"))
exclude_punct = string.punctuation

# ---------------------- Preprocessing Functions ----------------------

def remove_html(text):
    """Remove HTML tags from text."""
    return re.sub(r"<.*?>", "", text)

def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"(?:http|https|ftp)://\S+", "", text)

def convert_emojis(text):
    """Convert emojis to text descriptions."""
    return emoji.demojize(text)

def remove_punctuation(text):
    """Remove punctuation from text."""
    return text.translate(str.maketrans("", "", exclude_punct))

def remove_digits(text):
    """Remove digits from text and replace with spaces."""
    return ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])

def remove_stopwords(text):
    """Remove English stopwords."""
    words = nltk.word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words])

def preprocess_text(text):
    """Apply all preprocessing steps to the input text."""
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = convert_emojis(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = remove_stopwords(text)
    return text

# ---------------------- Sentiment Function ----------------------

def get_sentiment(text):
    """Return sentiment category using TextBlob polarity score."""
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return "positive"
        elif polarity < 0:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        print(f"Error processing text: {text}")
        return "error"

# ---------------------- Main Script ----------------------

def main():
    # Load and sample dataset
    df = pd.read_csv("SentimentAnalysisDS.csv")
    
    # Check for missing values and handle them
    df['review'] = df['review'].fillna('')  # Replace NaNs with empty strings
    
    # Sample 100 random reviews for processing
    df = df.sample(100).reset_index(drop=True)

    # Preprocess reviews
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # Get sentiment labels
    df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

    # Print sample output
    print(df[['review', 'cleaned_review', 'sentiment']].head())

if __name__ == "__main__":
    main()

