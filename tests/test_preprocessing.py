import pandas as pd
from src.preprocessing import *
import pandas as pd
from src.preprocessing import one_hot_encode


def test_one_hot_encode():

    df = pd.DataFrame({
        "MultipleLines": ["Yes", "No"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaymentMethod": ["Electronic check", "Mailed check"]
    })

    df = one_hot_encode(df)

    assert "Contract_One year" in df.columns
def test_label_encode():

    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "No"],
        "PaperlessBilling": ["Yes", "No"],
        "Churn": ["Yes", "No"]
    })

    df_encoded, encoders = label_encode(df)

    assert df_encoded["gender"].dtype == int
    assert len(encoders) == 6



import pandas as pd
from src.preprocessing import add_avg_monthly_spend

def test_avg_monthly_spend():

    df = pd.DataFrame({
        "tenure":[10],
        "TotalCharges":[500]
    })

    df = add_avg_monthly_spend(df)

    assert df["AvgMonthlySpend"][0] == 50

import pandas as pd
from src.preprocessing import split_data

def test_split_data():

    df = pd.DataFrame({
        "A": range(100),
        "Churn": [0]*80 + [1]*20
    })

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
