"""
Customer Review Sentiment Analysis
Standalone execution script adapted from sentiment_analysis.ipynb
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Ensure required NLTK data is downloaded
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set visual styling
sns.set_theme(style='whitegrid')

def main():
    dataset_path = 'reviews.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found. Ensure reviews.csv is in the working directory.")

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, index_col=0)

    # Data Cleaning: Remove rows missing review text
    initial_len = len(df)
    df.dropna(subset=['Review Text'], inplace=True)
    print(f"Dropped {initial_len - len(df)} rows with missing reviews. Total clean rows: {len(df)}")

    # Subset required columns
    df = df[['Review Text', 'Rating', 'Recommended IND']].copy()

    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    print("Computing VADER sentiment scores for customer reviews...")
    df['Sentiment Score'] = df['Review Text'].apply(lambda review: sia.polarity_scores(review)['compound'])

    # Classify sentiment into Positive, Negative, and Neutral
    def classify_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['Sentiment'] = df['Sentiment Score'].apply(classify_sentiment)

    print("\nSample Sentiment Predictions:")
    print(df[['Review Text', 'Rating', 'Sentiment Score', 'Sentiment']].head())

    print("\nSentiment Distribution:")
    distribution = df['Sentiment'].value_counts()
    print(distribution)

    # 1. Save distribution plot
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'], palette=['#2ecc71', '#95a5a6', '#e74c3c'])
    plt.title('Customer Review Sentiment Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    chart_path = 'sentiment_distribution_plot.png'
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved sentiment distribution plot to '{chart_path}'")

    # 2. Generate Word Clouds
    stop_words = set(stopwords.words('english'))
    # Common domain filler words
    stop_words.update(['dress', 'shirt', 'top', 'size', 'wear', 'look', 'bought'])

    # Positive Word Cloud
    positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Review Text'])
    wc_pos = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, min_font_size=10).generate(positive_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title('Key Themes in Positive Reviews')
    plt.tight_layout()
    plt.savefig('positive_word_cloud.png', dpi=300)
    plt.close()

    # Negative Word Cloud
    negative_text = ' '.join(df[df['Sentiment'] == 'Negative']['Review Text'])
    wc_neg = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, min_font_size=10).generate(negative_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title('Key Themes in Negative Reviews')
    plt.tight_layout()
    plt.savefig('negative_word_cloud.png', dpi=300)
    plt.close()

    print("Generated Word Clouds saved as 'positive_word_cloud.png' and 'negative_word_cloud.png'.")
    print("Project 2 execution finished successfully!")

if __name__ == '__main__':
    main()
