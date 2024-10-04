import streamlit as st
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load spaCy model
Nlp = spacy.load("en_core_web_sm")


# Function to process text
def process_text(text):
    # Tokenize article
    Tokens = word_tokenize(text)

    # Convert word to lowercase
    lower_tokens = [token.lower() for token in Tokens]

    # Remove Punctuation
    no_punct = [char for char in lower_tokens if char not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    article = [word for word in no_punct if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatize_word = [lemmatizer.lemmatize(word) for word in article]

    return lemmatize_word


# Function to perform sentiment analysis
def sentiment_analysis(text):
    SIA = SentimentIntensityAnalyzer()
    sentiment_scores = SIA.polarity_scores(''.join(text))
    return sentiment_scores


# Function to perform Name Entity Recognition
def ner(text):
    news_article = Nlp(' '.join(text))
    entities = [(ent.text, ent.label_) for ent in news_article.ents]
    return entities


# Streamlit app
st.title("News Article Analysis")

# Text input or file uploader
news_article = st.text_area("Paste your news article here:", height=250)

# Button to trigger analysis
if st.button("Analyze"):
    st.write("Sentiment Analysis")

    # Process text
    lemmatize_word = process_text(news_article)

    # Sentiment Analysis
    sentiment_scores = sentiment_analysis(lemmatize_word)

    # Sentiment Visualization
    labels = ['Positive', 'Negative', 'Neutral', 'Compound']
    values = [sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['compound']]

    fig_sentiment = plt.figure(figsize=(4, 3))
    plt.bar(labels, values)
    plt.xlabel('Sentiment')
    plt.ylabel('Scores')
    plt.title('Sentiment Analysis')
    st.pyplot(fig_sentiment)
    plt.close()  # Clear plot state

    # NER
    entities = ner(lemmatize_word)

    # Group entities by label
    entity_dict = {}
    for entity, label in entities:
        if label not in entity_dict:
            entity_dict[label] = [entity]
        else:
            entity_dict[label].append(entity)

    # Create a DataFrame
    data = []
    for label, ents in entity_dict.items():
        data.append({'Label': label, 'Entities': ', '.join(ents)})
    df = pd.DataFrame(data)

    # Display the DataFrame
    st.write("Named Entities:")
    st.table(df)

    # POS Tagging
    pos_tags = pos_tag(lemmatize_word)
    pos_freq = Counter(tag for word, tag in pos_tags)

    # Map Penn Treebank tags to simpler POS tags
    pos_map = {
        'NN': 'Noun',
        'VB': 'Verb',
        'JJ': 'Adjective',
        'RB': 'Adverb',
        'PRP': 'Pronoun'
        }

    # Simplify POS tags and count frequency
    simplified_pos_freq = {}
    for pos, freq in pos_freq.items():
        for key, value in pos_map.items():
            if pos.startswith(key):
                simplified_pos_freq[value] = simplified_pos_freq.get(value, 0) + freq

    # Display POS frequency
    st.write("Parts of Speech Frequency:")
    pos_labels = list(simplified_pos_freq.keys())
    pos_values = list(simplified_pos_freq.values())

    fig_pos = plt.figure(figsize=(4, 3))
    plt.bar(pos_labels, pos_values)
    plt.xlabel('Part of Speech')
    plt.ylabel('Frequency')
    plt.title('POS Frequency')
    st.pyplot(fig_pos)
    plt.close()  # Clear plot state
