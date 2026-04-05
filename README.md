# NLP Sentiment Analysis App

A Streamlit web app that performs **Natural Language Processing** and **Sentiment Analysis** on any text input.

## What This App Does

| Section | What it shows |
|---|---|
| What is NLP? | Plain-English explanation of all NLP concepts |
| Text Preprocessing Pipeline | Step-by-step cleaning: lowercase → tokenize → remove punctuation → stop words → lemmatize |
| Sentiment Analysis | Analyze any text with TextBlob + NLTK VADER |
| Compare Two Texts | Side-by-side sentiment comparison |

## NLP Pipeline Steps

```
Raw Text
   ↓ Lowercase
   ↓ Tokenize (split into words)
   ↓ Remove Punctuation
   ↓ Remove Stop Words (the, is, at...)
   ↓ Lemmatize (running → run)
   ↓ Expand Acronyms (nlp → natural language processing)
Final Clean Text → Sentiment Analysis
```

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
nlp_app/
├── app.py            ← Streamlit web app
├── model.py          ← NLP pipeline + sentiment logic
├── utils.py          ← Charts and visualizations
├── requirements.txt  ← Dependencies
└── README.md         ← This file
```

## Key Libraries

- **NLTK** - tokenization, stop words, lemmatization, VADER sentiment
- **TextBlob** - simple sentiment polarity and subjectivity
- **Streamlit** - web app framework

