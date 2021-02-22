import logging

from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from utils import timer


def baseline_model():
    clf = Pipeline([('scaler', StandardScaler()), ('svc', LinearSVC(C=1, dual=False))])
    return clf


def stacked_model(C=1, alpha=1, max_depth=None, n_neighbors=5, learning_rate=0.1, n_estimators=100, max_depth_gb=3):
    estimators = [
        ('LSVC', make_pipeline(StandardScaler(), LinearSVC(C=C, dual=False, max_iter=7345))),
        ('KNN', KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)),
        ('NaiveBayes', make_pipeline(StandardScaler(), BernoulliNB(alpha=alpha))),
        ('DecisionTree', DecisionTreeClassifier(max_depth=max_depth, random_state=42)),
        ('SVC', make_pipeline(StandardScaler(), SVC(C=C, kernel='rbf', gamma='scale', random_state=42)))
    ]
    final_estimators = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                                  max_depth=max_depth_gb, random_state=42)
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimators
    )
    return clf


def voting_model(C=1, alpha=1, max_depth=None, n_neighbors=5):
    estimators = [
        ('LSVC', make_pipeline(StandardScaler(), LinearSVC(C=C, dual=False, max_iter=7345))),
        ('KNN', KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)),
        ('NaiveBayes', make_pipeline(StandardScaler(), BernoulliNB(alpha=alpha))),
        ('DecisionTree', DecisionTreeClassifier(max_depth=max_depth, random_state=42)),
        ('SVC', make_pipeline(StandardScaler(), SVC(C=C, kernel='rbf', gamma='scale', random_state=42)))
    ]
    clf = VotingClassifier(estimators, voting='hard', n_jobs=-1)
    return clf

@timer
def optimize(model, distributions, X_train, y_train, n_iter, save=None):
    grid = RandomizedSearchCV(estimator=model, param_distributions=distributions, n_iter=n_iter,
                              n_jobs=-1, scoring='accuracy', random_state=42, cv=5, verbose=1)
    grid.fit(X_train, y_train)
    logging.info(f"{grid.best_estimator_.__class__.__name__} accuracy on train"
                 f" set: {accuracy_score(y_train, grid.predict(X_train))}")
    if save is not None:
        dump(grid.best_estimator_, save + '.joblib')
    return grid.best_estimator_

@timer
def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    score = accuracy_score(y_train, model.predict(X_train))
    logging.info(f"{model.__class__.__name__} accuracy on train set: {score}")
    return model, score

@timer
def evaluate(model, X_test, y_test):
    score = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"{model.__class__.__name__} accuracy on test set: {score}")
    return score
