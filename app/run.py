import sys
import json
import plotly
import joblib
import pandas as pd
from sqlalchemy import create_engine

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter


app = Flask(__name__)
sys.path.append('..')
import re
from models.train_classifier import tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

print('Creating charts...')
# Chart 1: Distribution of message genres
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

# Chart 2: Top Tokens in Sampled Dataset
def tokenizeWithoutStopWordsAndPunctuation(txt):
    txt = re.sub(r'[^a-zA-Z0-9]', ' ', txt)
    tokens = tokenize(txt)
    return [t for t in tokens if t not in stopwords.words('english')]
X, Y = df['message'], df.iloc[:,4:]
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# Sample dataset to reduce runtime
tokens = X_test.sample(5000).apply(tokenizeWithoutStopWordsAndPunctuation).explode()
top_tokens = tokens.value_counts(ascending=True, normalize=True).tail(10)

# Chart 3: Top categories
categories_count = Y_test.sum()
top_categories = categories_count.sort_values().tail(10)

# Chart 4: F1 Score x Percentage of Positive Classes
Y_pred = model.predict(X_test)
f1_scores = {'category': [], 'positive': [], 'f1_score': []}
# Get F1 Score for each category
for i, category in enumerate(Y_test.columns):
    # Drop classification 2
    bin = pd.concat([Y_test.iloc[:,i].reset_index(drop=True), pd.Series(Y_pred[:,i])], axis=1)
    bin.columns = ['true', 'pred']
    bin = bin[(bin['true']!=2) & (bin['pred']!=2)]
    
    # Skip categories with no true/predicted labels
    if (any(bin.sum() == 0)):
        continue

    f1_scores['category'].append(category)
    f1_scores['positive'].append(bin['true'].mean())
    f1_scores['f1_score'].append(f1_score(bin['true'], bin['pred']))

# create visuals
graphs = [
    { # Chart 1
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts
            )
        ],

        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    },

    { # Chart 2
        'data': [
            Bar(
                x=top_tokens.values,
                y=top_tokens.index,
                orientation='h'
            )
        ],

        'layout': {
            'title': 'Top Tokens in Dataset',
            'yaxis': {
                'title': "Token"
            },
            'xaxis': {
                'title': "Frequency"
            }
        }
    },

    { # Chart 3
        'data': [
            Bar(
                x=top_categories.values,
                y=top_categories.index,
                orientation='h'
            )
        ],

        'layout': {
            'title': 'Top Positive Categories',
            'yaxis': {
                'title': "Category"
            },
            'xaxis': {
                'title': "Quantity of Positive Samples"
            }
        }
    },

    { # Chart 4
        'data': [
            Scatter(
                x=f1_scores['positive'],
                y=f1_scores['f1_score'],
                text=f1_scores['category'],
                mode='markers'
            )
        ],

        'layout': {
            'title': 'Influence of Percentage of Positive Samples on F1 Score',
            'yaxis': {
                'title': "F1 Score"
            },
            'xaxis': {
                'title': "Percentage of Positive Samples"
            }
        }
    }
]

# encode plotly graphs in JSON
ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
print('OK')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=ids,
        graphJSON=graphJSON
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()