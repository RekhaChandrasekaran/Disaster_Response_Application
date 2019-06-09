import sys
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine

# NLTK
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_pattern, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
        
    normalized_tokens = [lemmatizer.lemmatize(token).lower().strip() 
                       for token in tokens]
        
    # remove stopwords
    STOPWORDS_EN = list(set(stopwords.words('english')))
        
    clean_tokens = [token for token in normalized_tokens if token not in STOPWORDS_EN]
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=6)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    reports = []
    
    for i in range(len(category_names)):
        reports.append([
            f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
            precision_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
            recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro')
        ])
    reports_df = pd.DataFrame(reports, columns=['f1_score', 'precision', 'recall'], index=category_names)
    reports_df.to_csv('report_randomforest.csv', index=False)
    print(reports_df)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()