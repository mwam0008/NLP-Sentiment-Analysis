"""
utils.py - Visualization helpers for NLP Sentiment Analysis App
"""

import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)


def plot_sentiment_gauge(score: float, title: str = "Sentiment Score"):
    """Plot a horizontal gauge bar showing sentiment from -1 to +1."""
    try:
        fig, ax = plt.subplots(figsize=(8, 2))

        # Background bar (grey)
        ax.barh(0, 2, left=-1, color='#EEEEEE', height=0.5)

        # Colored fill based on score
        color = '#4CAF50' if score >= 0.05 else '#F44336' if score <= -0.05 else '#FFC107'
        ax.barh(0, score, color=color, height=0.5, alpha=0.85)

        # Center line
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')

        ax.set_xlim(-1, 1)
        ax.set_yticks([])
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(['Very\nNegative', 'Negative', 'Neutral', 'Positive', 'Very\nPositive'])
        ax.set_title(f"{title}: {score:.4f}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Gauge plot error: {e}")
        raise


def plot_vader_breakdown(vader_scores: dict):
    """Bar chart showing VADER positive, negative, neutral breakdown."""
    try:
        categories = ['Positive', 'Negative', 'Neutral']
        values = [vader_scores['positive'], vader_scores['negative'], vader_scores['neutral']]
        colors = ['#4CAF50', '#F44336', '#FFC107']

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2%}', ha='center', fontweight='bold')

        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('VADER Sentiment Breakdown', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"VADER breakdown plot error: {e}")
        raise


def plot_token_count_comparison(steps: dict):
    """Bar chart comparing token count at each processing step."""
    try:
        step_names = []
        counts = []

        if 'tokens' in steps:
            step_names.append('After\nTokenize')
            counts.append(len(steps['tokens']))
        if 'no_punctuation' in steps:
            step_names.append('Remove\nPunctuation')
            counts.append(len(steps['no_punctuation']))
        if 'no_stopwords' in steps:
            step_names.append('Remove\nStop Words')
            counts.append(len(steps['no_stopwords']))
        if 'lemmatized' in steps:
            step_names.append('After\nLemmatize')
            counts.append(len(steps['lemmatized']))
        elif 'stemmed' in steps:
            step_names.append('After\nStemming')
            counts.append(len(steps['stemmed']))

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(step_names)))
        bars = ax.bar(step_names, counts, color=colors, edgecolor='white')

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(count), ha='center', fontweight='bold')

        ax.set_ylabel('Number of Tokens')
        ax.set_title('Token Count After Each Processing Step', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Token comparison plot error: {e}")
        raise


def plot_top_words(tokens: list, n: int = 15):
    """Horizontal bar chart of most frequent words."""
    try:
        counter = Counter(tokens)
        most_common = counter.most_common(n)

        if not most_common:
            return None

        words, counts = zip(*most_common)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))
        ax.barh(list(words)[::-1], list(counts)[::-1], color=colors[::-1])
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {n} Most Frequent Words', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Top words plot error: {e}")
        raise


def plot_textblob_scores(polarity: float, subjectivity: float):
    """Side-by-side gauges for TextBlob polarity and subjectivity."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))

        # Polarity gauge
        pol_color = '#4CAF50' if polarity >= 0.05 else '#F44336' if polarity <= -0.05 else '#FFC107'
        axes[0].barh(0, 2, left=-1, color='#EEEEEE', height=0.5)
        axes[0].barh(0, polarity, color=pol_color, height=0.5, alpha=0.85)
        axes[0].axvline(0, color='black', linewidth=1.2, linestyle='--')
        axes[0].set_xlim(-1, 1)
        axes[0].set_yticks([])
        axes[0].set_xticks([-1, 0, 1])
        axes[0].set_xticklabels(['Negative', 'Neutral', 'Positive'])
        axes[0].set_title(f'Polarity: {polarity:.4f}', fontweight='bold')

        # Subjectivity gauge
        sub_color = '#2196F3'
        axes[1].barh(0, 1, color='#EEEEEE', height=0.5)
        axes[1].barh(0, subjectivity, color=sub_color, height=0.5, alpha=0.85)
        axes[1].set_xlim(0, 1)
        axes[1].set_yticks([])
        axes[1].set_xticks([0, 0.5, 1])
        axes[1].set_xticklabels(['Objective', 'Mixed', 'Subjective'])
        axes[1].set_title(f'Subjectivity: {subjectivity:.4f}', fontweight='bold')

        plt.suptitle('TextBlob Analysis', fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"TextBlob score plot error: {e}")
        raise
