"""
model.py - All NLP logic for Sentiment Analysis App
Covers: cleaning, tokenization, stopword removal, stemming,
        lemmatization, and sentiment scoring (TextBlob + NLTK VADER)
"""

import logging
import string
import nltk
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download all required NLTK data once
def download_nltk_data():
    """Download all required NLTK datasets."""
    packages = ['punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon', 'averaged_perceptron_tagger']
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            logging.warning(f"Could not download {pkg}: {e}")

download_nltk_data()


# ── STEP 1: Text Cleaning ─────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert all text to lowercase."""
    try:
        result = text.lower()
        logging.info("Text converted to lowercase.")
        return result
    except Exception as e:
        logging.error(f"Lowercase conversion failed: {e}")
        raise


# ── STEP 2: Tokenization ──────────────────────────────────────

def tokenize(text: str) -> list:
    """Split text into individual word tokens."""
    try:
        tokens = nltk.word_tokenize(text)
        logging.info(f"Tokenized into {len(tokens)} tokens.")
        return tokens
    except Exception as e:
        logging.error(f"Tokenization failed: {e}")
        raise


# ── STEP 3: Remove Punctuation ───────────────────────────────

def remove_punctuation(tokens: list) -> list:
    """Remove all punctuation marks from token list."""
    try:
        result = [token for token in tokens if token not in string.punctuation]
        logging.info(f"Removed punctuation. {len(tokens) - len(result)} punctuation marks removed.")
        return result
    except Exception as e:
        logging.error(f"Punctuation removal failed: {e}")
        raise


# ── STEP 4: Remove Stop Words ────────────────────────────────

def remove_stopwords(tokens: list) -> list:
    """Remove common English stop words (the, is, at, etc.)."""
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        result = [token for token in tokens if token not in stop_words]
        logging.info(f"Removed stop words. {len(tokens) - len(result)} stop words removed.")
        return result
    except Exception as e:
        logging.error(f"Stop word removal failed: {e}")
        raise


# ── STEP 5a: Stemming ────────────────────────────────────────

def stem_tokens(tokens: list) -> list:
    """Reduce words to their root form using Porter Stemmer.
    e.g. 'running' → 'run', 'happiness' → 'happi' (crude but fast)
    """
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        result = [stemmer.stem(token) for token in tokens]
        logging.info("Stemming complete.")
        return result
    except Exception as e:
        logging.error(f"Stemming failed: {e}")
        raise


# ── STEP 5b: Lemmatization ───────────────────────────────────

def lemmatize_tokens(tokens: list) -> list:
    """Reduce words to dictionary base form using WordNet.
    e.g. 'running' → 'run', 'better' → 'good' (smarter than stemming)
    """
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        result = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
        logging.info("Lemmatization complete.")
        return result
    except Exception as e:
        logging.error(f"Lemmatization failed: {e}")
        raise


# ── STEP 6: Expand Acronyms ──────────────────────────────────

def expand_acronyms(tokens: list, custom_dict: dict = None) -> list:
    """Replace acronyms/slang with full words using a dictionary."""
    try:
        default_dict = {
            'nlp': 'natural language processing',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nn': 'neural network',
        }
        if custom_dict:
            default_dict.update(custom_dict)
        result = [default_dict.get(token, token) for token in tokens]
        logging.info("Acronym expansion complete.")
        return result
    except Exception as e:
        logging.error(f"Acronym expansion failed: {e}")
        raise


# ── STEP 7: Full Pipeline ────────────────────────────────────

def full_pipeline(text: str, use_stemming=False, expand=True) -> dict:
    """
    Run the complete NLP preprocessing pipeline on a piece of text.
    Returns a dict with all intermediate steps for display.
    """
    try:
        steps = {}
        steps['original'] = text

        lowered = to_lowercase(text)
        steps['lowercase'] = lowered

        tokens = tokenize(lowered)
        steps['tokens'] = tokens

        no_punct = remove_punctuation(tokens)
        steps['no_punctuation'] = no_punct

        no_stopwords = remove_stopwords(no_punct)
        steps['no_stopwords'] = no_stopwords

        if use_stemming:
            processed = stem_tokens(no_stopwords)
            steps['stemmed'] = processed
        else:
            processed = lemmatize_tokens(no_stopwords)
            steps['lemmatized'] = processed

        if expand:
            processed = expand_acronyms(processed)
            steps['expanded'] = processed

        steps['final_tokens'] = processed
        steps['final_text'] = ' '.join(processed)

        logging.info("Full pipeline complete.")
        return steps
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


# ── STEP 8: Sentiment Analysis ───────────────────────────────

def analyze_textblob(text: str) -> dict:
    """Analyze sentiment using TextBlob.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    try:
        from textblob import TextBlob
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        if polarity >= 0.05:
            label = "Positive 😊"
        elif polarity <= -0.05:
            label = "Negative 😞"
        else:
            label = "Neutral 😐"

        logging.info(f"TextBlob sentiment: {label} ({polarity:.3f})")
        return {
            'polarity': round(polarity, 4),
            'subjectivity': round(subjectivity, 4),
            'label': label
        }
    except Exception as e:
        logging.error(f"TextBlob analysis failed: {e}")
        raise


def analyze_vader(text: str) -> dict:
    """Analyze sentiment using NLTK VADER.
    Returns compound score and individual pos/neg/neu scores.
    """
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            label = "Positive 😊"
        elif compound <= -0.05:
            label = "Negative 😞"
        else:
            label = "Neutral 😐"

        logging.info(f"VADER sentiment: {label} ({compound:.3f})")
        return {
            'compound': round(compound, 4),
            'positive': round(scores['pos'], 4),
            'negative': round(scores['neg'], 4),
            'neutral': round(scores['neu'], 4),
            'label': label
        }
    except Exception as e:
        logging.error(f"VADER analysis failed: {e}")
        raise
