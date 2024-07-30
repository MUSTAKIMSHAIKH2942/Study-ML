from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import plotly.express as px
import json
import logging
import plotly

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_decision_tree(X, y):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    return model

def train_svr(X, y):
    model = SVR()
    model.fit(X, y)
    return model

def train_knn(X, y):
    model = KNeighborsRegressor()
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def train_lstm(X, y):
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(1, X.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

ALGORITHMS = {
    'linear_regression': train_linear_regression,
    'decision_tree': train_decision_tree,
    'svr': train_svr,
    'knn': train_knn,
    'random_forest': train_random_forest,
    'lstm': train_lstm
}

EXPLANATIONS = {
    'linear_regression': {
        'description': 'Linear Regression is a simple algorithm that assumes a linear relationship between input and output variables.',
        'key-points': [
            'Assumes linear relationship between variables.',
            'Sensitive to outliers.',
            'Easy to interpret.'
        ]
    },
    'decision_tree': {
        'description': 'Decision Tree is a non-parametric supervised learning method used for classification and regression.',
        'key-points': [
            'Non-linear model.',
            'Easy to interpret.',
            'Prone to overfitting.'
        ]
    },
    'svr': {
        'description': 'Support Vector Regression (SVR) uses the same principles as Support Vector Machines for classification, with minor changes to make it suitable for regression.',
        'key-points': [
            'Effective in high-dimensional spaces.',
            'Uses kernel functions.',
            'Sensitive to parameter selection.'
        ]
    },
    'knn': {
        'description': 'K-Nearest Neighbors (KNN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.',
        'key-points': [
            'Simple and intuitive.',
            'No training phase.',
            'Computationally expensive at prediction time.'
        ]
    },
    'random_forest': {
        'description': 'Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.',
        'key-points': [
            'Reduces overfitting.',
            'Handles large datasets well.',
            'Can handle missing values.'
        ]
    },
    'lstm': {
        'description': 'Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.',
        'key-points': [
            'Good for sequential data.',
            'Can handle long-term dependencies.',
            'Complex and computationally intensive.'
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html', explanations=EXPLANATIONS)

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        X = np.array(data['X']).reshape(-1, 1)
        y = np.array(data['y'])
        algorithm = data['algorithm']

        model = ALGORITHMS[algorithm](X, y)
        y_pred = model.predict(X)

        fig = px.scatter(x=X.flatten(), y=y, title=algorithm.replace('_', ' ').title(), labels={'x':'X', 'y':'y'})
        fig.add_scatter(x=X.flatten(), y=y_pred, mode='lines', name='Prediction')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get explanation and key points
        explanation = EXPLANATIONS[algorithm]

        return jsonify({
            'graphJSON': graphJSON,
            'description': explanation['description'],
            'key_points': explanation['key-points']
        })
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
