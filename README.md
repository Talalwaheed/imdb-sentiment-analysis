# IMDB Sentiment Analysis

A Natural Language Processing project that reads IMDB movie reviews and determines 
whether the sentiment expressed is positive or negative. Built using classical NLP 
techniques and a Naive Bayes classifier trained on 50,000 reviews.

## Overview

Sentiment analysis is one of the most widely applied NLP tasks in the industry, 
used everywhere from product reviews to social media monitoring. This project 
implements a full text classification pipeline — from raw review text to a 
deployable model — and includes an interactive predictor where you can test 
your own movie reviews in real time.

## What This Project Does

- Loads and validates the IMDB dataset with robust encoding detection
- Removes duplicates and handles missing values automatically
- Cleans review text by stripping HTML tags, URLs, punctuation, and stopwords
- Vectorizes text using TF-IDF with a vocabulary of 5,000 features
- Trains a Multinomial Naive Bayes classifier on an 80/20 train-test split
- Evaluates the model using accuracy, precision, recall, and F1 score
- Visualizes results through a confusion matrix and sentiment distribution charts
- Identifies the top 20 most indicative words for positive and negative classes
- Includes an interactive widget to predict sentiment on custom review input
- Saves the trained model and vectorizer as pickle files for future use

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| Precision | High |
| Recall | Balanced |
| F1 Score | Consistent across both classes |

## Key Features

- Fully automated text cleaning pipeline
- Interactive sentiment predictor built with ipywidgets
- Top word analysis showing which terms drive positive vs negative predictions
- Model and vectorizer exported as reusable pickle files

## Tech Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (TfidfVectorizer, MultinomialNB)
- NLTK (stopwords)
- ipywidgets (interactive predictor)
- Google Colab

## Dataset

The project uses the IMDB Dataset containing 50,000 labeled movie reviews evenly 
split between positive and negative sentiment.

## Output Files

| File | Description |
|------|-------------|
| imdb_mnb_model.pkl | Trained Naive Bayes model |
| tfidf_vectorizer.pkl | Fitted TF-IDF vectorizer |

## Author

M. Talal Bin Waheed
