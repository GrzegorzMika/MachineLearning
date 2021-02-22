import logging

import pandas as pd
import numpy as np
from scipy.stats import uniform
from sklearn.model_selection import train_test_split

from FakeNewsData import load_and_clean, preprocess_data
from FakeNewsModel import baseline_model, stacked_model, train, evaluate, optimize, voting_model

PREPROCESS_DATA = False

logging.basicConfig(level=logging.INFO)

if PREPROCESS_DATA:
    data = load_and_clean('corona_fake.csv')
    data = preprocess_data(data)
    data.to_csv('corona_fake_preprocessed.csv', index=False)
else:
    data = pd.read_csv('corona_fake_preprocessed.csv')

X, y = data.drop(['title', 'text', 'source', 'label', 'text_standard'], axis=1).astype(np.float64), \
       data['label'].replace({'TRUE': 1, 'FAKE': 0}).astype(np.float64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf = baseline_model()
train(clf, X_train, y_train)
evaluate(clf, X_test, y_test)

stacked_model_params = dict(
    LSVC__linearsvc__C=[i for i in range(1, 100)],
    SVC__svc__C=[i for i in range(1, 100)],
    NaiveBayes__bernoullinb__alpha=uniform(loc=0, scale=10),
    DecisionTree__max_depth=[i for i in range(1, 20)],
    KNN__n_neighbors=[i for i in range(1, 50)],
    final_estimator__learning_rate=uniform(loc=0.01, scale=1),
    final_estimator__n_estimators=[50, 100, 200, 500, 1000],
    final_estimator__max_depth=[i for i in range(1, 50)]
)
clf = stacked_model()
clf = optimize(clf, stacked_model_params, X_train, y_train, n_iter=100)
evaluate(clf, X_test, y_test)

voting_model_params = dict(
    LSVC__linearsvc__C=[i for i in range(1, 100)],
    SVC__svc__C=[i for i in range(1, 100)],
    NaiveBayes__bernoullinb__alpha=uniform(loc=0, scale=10),
    DecisionTree__max_depth=[i for i in range(1, 20)],
    KNN__n_neighbors=[i for i in range(1, 50)],
)
clf = voting_model()
clf = optimize(clf, voting_model_params, X_train, y_train, n_iter=1500)
evaluate(clf, X_test, y_test)
