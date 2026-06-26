import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
def reg_metric(X_test,y_test,y_pred):
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_test, y_pred)
    
    n = len(y_test)
    p = X_test.shape[1]
    
    adj_r2 = 1 - (
        (1-r2)*(n-1)/(n-p-1)
    )
    
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)
    print("Adjusted R² :", adj_r2)
    return mae,mse,rmse,r2,adj_r2

def train_linear_regression(X_train, y_train,X_test):

    model = LinearRegression()

    start = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start
    y_pred = model.predict(X_test)

    return model, training_time,y_pred

def train_ridge(X_train, y_train,X_test):

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

    best_ridge=grid.fit(X_train, y_train)

    training_time = time.time() - start
    y_pred = best_ridge.predict(X_test)

    return grid.best_estimator_, training_time,y_pred

def train_lasso(X_train, y_train,X_test):

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
    y_pred = best_lasso.predict(X_test_scaled)

    return (
        best_lasso,
        scaler,
        X_train_scaled,
        X_test_scaled,
        training_time,
        y_pred
    )
def train_elasticnet(X_train, y_train,X_test):



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

    grid.fit(X_train, y_train)      # ← Fit first

    best_elastic = grid.best_estimator_   # ← Then get best model

    training_time = time.time() - start

    y_pred = best_elastic.predict(X_test)

    return best_elastic, training_time, y_pred

def train_decision_tree_regressor(X_train, y_train,X_test):

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

    best_dt=grid.fit(X_train, y_train)

    training_time = time.time() - start
    y_pred = best_dt.predict(X_test)
    return grid.best_estimator_, training_time,y_pred

def train_random_forest_regressor(X_train, y_train,X_test):

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
    best_rf=grid.best_estimator_
    y_pred = best_rf.predict(X_test)
    
    return best_rf, training_time,y_pred

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
    y_pred = best_svr.predict(X_test_scaled)
    return (
        best_svr,
        scaler,
        training_time,
        y_pred
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
import matplotlib.pyplot as plt
def act_vs_pred(y_test,y_pred,a):
    plt.figure(figsize=(6,5))

    plt.scatter(
        y_test,
        y_pred
    )
    
    plt.xlabel("Actual MonthlyCharges")
    plt.ylabel("Predicted MonthlyCharges")
    
    plt.title(f"{a}: Actual vs Predicted")
    
    plt.show()

def residual_distribution(y_test,y_pred):
    residuals = y_test - y_pred
    
    plt.figure(figsize=(6,5))
    
    plt.hist(
        residuals,
        bins=30
    )
    
    plt.title("Residual Distribution")
    
    plt.xlabel("Residuals")
    
    plt.show()
