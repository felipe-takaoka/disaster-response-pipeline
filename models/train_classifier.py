import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Loads messages data from the provided database returning the original messages, category values and category names.

    Parameters:
    database_filepath (str): filepath for the database

    Returns:
    pandas.Series: Series containing text messages
    pandas.DataFrame: DataFrame with binary data indicating message categories in columns
    List[str]: List of category names
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
    Returns the tokens of a text string. The text is pre-processed to replace URLs with a placeholder.
    The tokens are lemmatized, converted to lower case and stripped of leading and trailing white spaces.

    Parameters:
    text (str): raw text to be tokenized

    Returns:
    List[str]: list of the processed tokens of text
    """
    
    # Replace URLs with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    re.sub(url_regex, 'urlplaceholder', text)
    
    # Tokenize by words
    tokens = word_tokenize(text)
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """
    Builds the multi-target model for classifying messages with their categories.
    The model is composed of a Pipeline that creates TFIDF and verb counts as features.
    It uses Grid Search Cross Validation for hyper-parameter optimization.
    Finnaly the classifier consists of a MultiOutputClassifier with a RandomForestClassifier.

    Returns:
    sklearn.base.BaseEstimator: multi-target classifier of the text messages
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [10],#, 100, 500],
        'clf__estimator__max_depth': [None],#, 5, 10, 20],
        'clf__estimator__min_samples_split': [2]#, 10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model on the test set by printing the classification report for each category name.

    Parameters:
    model (sklearn.base.BaseEstimator): model to be evaluated
    X_test (pandas.DataFrame): feature values for the test set
    Y_test (pandas.DataFrame): target values for the test set
    category_names (List[str]): list of category names
    """
    
    Y_pred = model.predict(X_test)

    for i, cat in enumerate(category_names):
        print(cat)
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    """Saves model as a pickle file in model_filepath"""
    joblib.dump(model, model_filepath)


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