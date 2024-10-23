import mlflow
import pandas as pd
import requests
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import dagshub

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])
print (df.head())

# data preprocessing

# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

df = normalize_text(df)
df['sentiment'].value_counts()

#Keeing just 2 sentiments
x = df['sentiment'].isin(['happiness','sadness'])
df = df[x]

df['sentiment'] = df['sentiment'].replace({'sadness':0, 'happiness':1})
print (df.head())

vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dagshub.init(repo_owner='shishant1984', repo_name='MLops_project', mlflow=True )
mlflow.set_tracking_uri('https://dagshub.com/shishant1984/MLops_project.mlflow')

mlflow.set_experiment("Logistic Regression Baseline")


with mlflow.start_run():
    # Log preprocessing parameters
    mlflow.log_param("vectorizer", "Bag of Words")
    mlflow.log_param("num_features", 1000)
    mlflow.log_param("test_size", 0.2)
    
    # Model building and training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Log model parameters
    mlflow.log_param("model", "Logistic Regression")
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")