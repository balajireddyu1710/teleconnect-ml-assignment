import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import seaborn as sns
def predictions_c(X_test,a):
    y_pred = a.predict(X_test)
    y_prob = a.predict_proba(X_test)[:, 1]
    return y_pred,y_prob

def classification_metrics(y_true, y_pred, y_prob):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    return accuracy, precision, recall, f1, roc_auc


def plot_confusion_matrix(y_true, y_pred,model_name):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    return cm

def get_classification_report(y_true, y_pred):

    return classification_report(y_true, y_pred)

def regression_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def plot_roc_curve(model, X_test, y_test, model_name):

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(f"{model_name} ROC Curve")
    plt.show()

def classification_results_table(results):

    return pd.DataFrame(results)