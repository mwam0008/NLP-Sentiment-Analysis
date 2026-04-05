"""
app.py - Streamlit Web App for NLP Sentiment Analysis
Covers: preprocessing pipeline + TextBlob + NLTK VADER sentiment
Run with: streamlit run app.py
"""

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from model import (
    full_pipeline,
    analyze_textblob,
    analyze_vader,
    to_lowercase,
    tokenize,
    remove_punctuation,
    remove_stopwords,
    stem_tokens,
    lemmatize_tokens,
)
from utils import (
    plot_sentiment_gauge,
    plot_vader_breakdown,
    plot_token_count_comparison,
    plot_top_words,
    plot_textblob_scores,
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 NLP Sentiment Analysis")
st.markdown("Analyze text using **Natural Language Processing** — clean it, process it, and detect its sentiment.")

# ── Sidebar Navigation ────────────────────────────────────────
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio("Choose a section:", [
    "📖 What is NLP?",
    "🧹 Text Preprocessing Pipeline",
    "😊 Sentiment Analysis",
    "🔬 Compare Two Texts",
])

# ════════════════════════════════════════════════════════════
# SECTION 1 — What is NLP?
# ════════════════════════════════════════════════════════════
if section == "📖 What is NLP?":
    st.header("📖 What is NLP?")

    st.markdown("""
    **Natural Language Processing (NLP)** is how computers understand and work with human language.

    ---

    ### The NLP Pipeline (what we do to raw text before analysis):

    | Step | What it does | Example |
    |---|---|---|
    | **Lowercase** | Make everything small caps | `"Hello"` → `"hello"` |
    | **Tokenize** | Split into individual words | `"I love NLP"` → `["I", "love", "NLP"]` |
    | **Remove Punctuation** | Strip `! . , ? ;` etc. | `["hello!"]` → `["hello"]` |
    | **Remove Stop Words** | Remove filler words | Remove `"the"`, `"is"`, `"at"` |
    | **Stemming** | Chop word endings (crude) | `"running"` → `"run"`, `"happiness"` → `"happi"` |
    | **Lemmatization** | Find true base form (smart) | `"running"` → `"run"`, `"better"` → `"good"` |
    | **Expand Acronyms** | Replace slang/short forms | `"nlp"` → `"natural language processing"` |

    ---

    ### Sentiment Analysis:

    | Tool | How it works | Score range |
    |---|---|---|
    | **TextBlob** | Dictionary-based, simple | Polarity: -1 to +1 |
    | **VADER** | Built for social media text | Compound: -1 to +1 |

    > **Polarity:** -1 = very negative, 0 = neutral, +1 = very positive
    """)

    st.info("👈 Use the sidebar to try the **Text Preprocessing Pipeline** or **Sentiment Analysis** sections!")

# ════════════════════════════════════════════════════════════
# SECTION 2 — Text Preprocessing Pipeline
# ════════════════════════════════════════════════════════════
elif section == "🧹 Text Preprocessing Pipeline":
    st.header("🧹 Text Preprocessing Pipeline")
    st.markdown("Paste any text and watch it get cleaned step by step!")

    default_text = "Natural Language Processing (NLP) is making machines understand human language. It's an exciting field of AI and ML! Running, runs, and ran are all forms of 'run'."

    user_text = st.text_area("✏️ Enter your text here:", value=default_text, height=120)

    col1, col2 = st.columns(2)
    use_stemming = col1.checkbox("Use Stemming (faster, cruder)", value=False)
    use_lemma = col2.checkbox("Use Lemmatization (slower, smarter)", value=True)

    if use_stemming and use_lemma:
        st.warning("Both selected — will use Lemmatization.")
        use_stemming = False

    if st.button("🚀 Run Pipeline"):
        with st.spinner("Processing text..."):
            try:
                steps = full_pipeline(user_text, use_stemming=use_stemming)

                st.success("✅ Pipeline complete!")

                # Step-by-step display
                st.subheader("📋 Step-by-Step Results")

                with st.expander("Step 1 — Lowercase", expanded=True):
                    st.code(steps['lowercase'])

                with st.expander("Step 2 — Tokenized"):
                    st.write(steps['tokens'])
                    st.caption(f"Total tokens: {len(steps['tokens'])}")

                with st.expander("Step 3 — Remove Punctuation"):
                    st.write(steps['no_punctuation'])
                    st.caption(f"Tokens remaining: {len(steps['no_punctuation'])}")

                with st.expander("Step 4 — Remove Stop Words"):
                    st.write(steps['no_stopwords'])
                    st.caption(f"Tokens remaining: {len(steps['no_stopwords'])}")

                if 'stemmed' in steps:
                    with st.expander("Step 5 — Stemmed Tokens"):
                        st.write(steps['stemmed'])
                elif 'lemmatized' in steps:
                    with st.expander("Step 5 — Lemmatized Tokens"):
                        st.write(steps['lemmatized'])

                with st.expander("Step 6 — Expanded Acronyms (Final)"):
                    st.write(steps.get('expanded', steps['final_tokens']))

                st.subheader("📊 Token Count at Each Step")
                fig = plot_token_count_comparison(steps)
                st.pyplot(fig)

                st.subheader("🔤 Top Words in Your Text")
                fig2 = plot_top_words(steps['final_tokens'], n=10)
                if fig2:
                    st.pyplot(fig2)
                else:
                    st.info("Not enough words to show top words chart.")

                st.subheader("✅ Final Cleaned Text")
                st.success(steps['final_text'])

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 3 — Sentiment Analysis
# ════════════════════════════════════════════════════════════
elif section == "😊 Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    st.markdown("Enter any text and find out if it's **positive**, **negative**, or **neutral**.")

    examples = {
        "Custom (type your own)": "",
        "Positive example": "I absolutely love this product! It's amazing and works perfectly. Highly recommend!",
        "Negative example": "This is terrible. I wasted my money. Worst experience ever. Very disappointed.",
        "Neutral example": "The package arrived on Tuesday. It contained three items as described.",
        "Mixed example": "The food was great but the service was really slow and disappointing.",
    }

    selected = st.selectbox("Try an example or write your own:", list(examples.keys()))
    default = examples[selected]

    user_text = st.text_area("✏️ Your text:", value=default, height=120)

    run_preprocess = st.checkbox("Clean text before analyzing (recommended)", value=True)

    if st.button("🔍 Analyze Sentiment"):
        if not user_text.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Optionally preprocess
                    if run_preprocess:
                        steps = full_pipeline(user_text)
                        analysis_text = steps['final_text']
                        st.info(f"📝 Cleaned text used for analysis: *{analysis_text}*")
                    else:
                        analysis_text = user_text

                    # TextBlob analysis
                    st.subheader("🔵 TextBlob Analysis")
                    tb = analyze_textblob(analysis_text)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sentiment", tb['label'])
                    col2.metric("Polarity", f"{tb['polarity']:.4f}")
                    col3.metric("Subjectivity", f"{tb['subjectivity']:.4f}")

                    fig = plot_textblob_scores(tb['polarity'], tb['subjectivity'])
                    st.pyplot(fig)

                    st.markdown("""
                    > **Polarity:** -1 = very negative → 0 = neutral → +1 = very positive
                    > **Subjectivity:** 0 = factual/objective → 1 = opinion-based/subjective
                    """)

                    st.divider()

                    # VADER analysis
                    st.subheader("🟣 NLTK VADER Analysis")
                    vader = analyze_vader(analysis_text)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Sentiment", vader['label'])
                    col2.metric("Compound Score", f"{vader['compound']:.4f}")
                    col3.metric("Positive", f"{vader['positive']:.2%}")
                    col4.metric("Negative", f"{vader['negative']:.2%}")

                    fig2 = plot_sentiment_gauge(vader['compound'], "VADER Compound Score")
                    st.pyplot(fig2)

                    fig3 = plot_vader_breakdown(vader)
                    st.pyplot(fig3)

                    st.markdown("""
                    > **VADER** is specifically trained for social media and short text.
                    > Compound ≥ 0.05 = Positive | ≤ -0.05 = Negative | In between = Neutral
                    """)

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

# ════════════════════════════════════════════════════════════
# SECTION 4 — Compare Two Texts
# ════════════════════════════════════════════════════════════
elif section == "🔬 Compare Two Texts":
    st.header("🔬 Compare Sentiment of Two Texts")
    st.markdown("Enter two texts and compare their sentiment scores side by side.")

    col1, col2 = st.columns(2)
    text1 = col1.text_area("📝 Text 1:", value="I love this! It's amazing and wonderful.", height=100)
    text2 = col2.text_area("📝 Text 2:", value="I hate this. It's terrible and broken.", height=100)

    if st.button("⚖️ Compare"):
        if not text1.strip() or not text2.strip():
            st.warning("Please enter both texts!")
        else:
            with st.spinner("Analyzing both texts..."):
                try:
                    # Process both
                    steps1 = full_pipeline(text1)
                    steps2 = full_pipeline(text2)

                    tb1 = analyze_textblob(steps1['final_text'])
                    tb2 = analyze_textblob(steps2['final_text'])
                    v1 = analyze_vader(steps1['final_text'])
                    v2 = analyze_vader(steps2['final_text'])

                    st.subheader("📊 TextBlob Comparison")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Text 1**")
                        st.metric("Sentiment", tb1['label'])
                        st.metric("Polarity", f"{tb1['polarity']:.4f}")
                        st.metric("Subjectivity", f"{tb1['subjectivity']:.4f}")
                    with c2:
                        st.markdown("**Text 2**")
                        st.metric("Sentiment", tb2['label'])
                        st.metric("Polarity", f"{tb2['polarity']:.4f}")
                        st.metric("Subjectivity", f"{tb2['subjectivity']:.4f}")

                    st.divider()

                    st.subheader("📊 VADER Comparison")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Text 1**")
                        st.metric("Sentiment", v1['label'])
                        st.metric("Compound", f"{v1['compound']:.4f}")
                        fig = plot_sentiment_gauge(v1['compound'], "Text 1")
                        st.pyplot(fig)
                    with c2:
                        st.markdown("**Text 2**")
                        st.metric("Sentiment", v2['label'])
                        st.metric("Compound", f"{v2['compound']:.4f}")
                        fig = plot_sentiment_gauge(v2['compound'], "Text 2")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Comparison failed: {e}")

# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**CST2216 — Individual Term Project**")
st.sidebar.markdown("NLP Sentiment Analysis App")
