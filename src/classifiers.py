import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

def train_logistic(X_train, y_train):

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }

    model = LogisticRegression(
        max_iter=5000,
        solver='liblinear'
    )

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    best_model = grid.best_estimator_

    return best_model, training_time

def train_decision_tree(X_train, y_train):

    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    model = DecisionTreeClassifier(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    return grid.best_estimator_, training_time

def train_random_forest(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    return grid.best_estimator_, training_time

from sklearn.svm import SVC
import time

def train_svm(X_train, y_train):

    model = SVC(
        C=1,
        kernel='rbf',
        probability=True,
        class_weight='balanced'
    )

    start = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start

    return model, training_time


def train_knn(X_train, y_train):

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }

    model = KNeighborsClassifier()

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    return grid.best_estimator_, training_time

import pickle

def save_classifier(model, path):

    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_classifier(path):

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model
