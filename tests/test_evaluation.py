import pandas as pd
from sklearn.linear_model import LogisticRegression


from src.evaluation import predictions_c

def test_predictions():

    X = pd.DataFrame({
        "A":[1,2,3,4],
        "B":[4,3,2,1]
    })

    y = [0,1,0,1]

    model = LogisticRegression()

    model.fit(X,y)

    y_pred, y_prob = predictions_c(X, model)

    assert len(y_pred) == 4
    assert len(y_prob) == 4
from src.evaluation import classification_metrics

def test_classification_metrics():

    y_true = [0,1,1,0]
    y_pred = [0,1,0,0]
    y_prob = [0.2,0.8,0.4,0.1]

    accuracy, precision, recall, f1, roc_auc = classification_metrics(
        y_true,
        y_pred,
        y_prob
    )

    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
    assert 0 <= roc_auc <= 1
import numpy as np
from src.evaluation import plot_confusion_matrix

def test_confusion_matrix():

    y_true = [0,1,1,0]
    y_pred = [0,1,0,0]

    cm = plot_confusion_matrix(
        y_true,
        y_pred,
        "Test"
    )

    assert cm.shape == (2,2)

    assert np.array_equal(
        cm,
        np.array([[2,0],[1,1]])
    )
from src.evaluation import get_classification_report

def test_classification_report():

    y_true = [0,1,1,0]
    y_pred = [0,1,0,0]

    report = get_classification_report(
        y_true,
        y_pred
    )

    assert isinstance(report, str)

    assert "precision" in report
from src.evaluation import regression_metrics

def test_regression_metrics():

    y_true = [10,20,30]
    y_pred = [11,19,29]

    metrics = regression_metrics(
        y_true,
        y_pred
    )

    assert "MAE" in metrics
    assert "MSE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
import pandas as pd
from src.evaluation import classification_results_table

def test_results_table():

    results = [
        ["Logistic",0.8,0.7,0.8,0.75,0.85,0.12],
        ["Random Forest",0.82,0.76,0.81,0.78,0.88,0.35]
    ]

    df = classification_results_table(results)

    assert isinstance(df, pd.DataFrame)
