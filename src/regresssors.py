import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
def train_linear_regression(X_train, y_train):

    model = LinearRegression()

    start = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start

    return model, training_time

def train_ridge(X_train, y_train):

    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    return grid.best_estimator_, training_time

def train_lasso(X_train, y_train):

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
    }

    grid = GridSearchCV(
        estimator=Lasso(max_iter=5000),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_lasso = grid.best_estimator_

    start = time.time()

    best_lasso.fit(X_train_scaled, y_train)

    training_time = time.time() - start

    return (
        best_lasso,
        scaler,
        X_train_scaled,
        X_test_scaled,
        training_time
    )
def train_elasticnet(X_train, y_train):

    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1],
        'l1_ratio': [0.2, 0.5, 0.8]
    }

    grid = GridSearchCV(
        estimator=ElasticNet(max_iter=5000),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    return grid.best_estimator_, training_time

def train_decision_tree_regressor(X_train, y_train):

    param_grid = { 
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]            
        # Copy from notebook
    }

    grid = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    return grid.best_estimator_, training_time

def train_random_forest_regressor(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
        # Copy from notebook
    }

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train, y_train)

    training_time = time.time() - start

    return grid.best_estimator_, training_time

import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def train_svr(X_train, X_test, y_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [1, 10, 100],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    start = time.time()

    grid.fit(X_train_scaled, y_train)

    training_time = time.time() - start

    best_svr = grid.best_estimator_

    return (
        best_svr,
        scaler,
        training_time
    )

def save_regressor(model, path):

    with open(path, "wb") as f:
        pickle.dump(model, f)

    def load_regressor(path):

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model

def feature_importance(model, X):

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    return importance.sort_values(
        by="Importance",
        ascending=False
    )

def coefficient_importance(model, X):

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    })

    coef_df["AbsCoef"] = coef_df["Coefficient"].abs()

    return coef_df.sort_values(
        by="AbsCoef",
        ascending=False
    )