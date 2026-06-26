import pandas as pd
import numpy as np
#functions
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)


def label_encode(df):

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"
    ]

    encoders = {}

    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def one_hot_encode(df):

    multi_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    df = pd.get_dummies(
        df,
        columns=multi_cols,
        drop_first=True
    )

    return df

def add_avg_monthly_spend(df):

    df["AvgMonthlySpend"] = np.where(
        df['tenure'] == 0,
        0,
        df['TotalCharges'] / df['tenure']
    )

    return df

def add_service_count(df):

    service_cols = [
        'PhoneService',
        'OnlineSecurity_Yes',
        'OnlineBackup_Yes',
        'DeviceProtection_Yes',
        'TechSupport_Yes',
        'StreamingTV_Yes',
        'StreamingMovies_Yes'
    ]
    df['ServiceCount'] = df[service_cols].astype(int).sum(axis=1)
    df["ServiceCount"] = df[service_cols].sum(axis=1)

    return df

def add_contract_value(df):

    df["ContractValue"] = df['MonthlyCharges'] * df['tenure']

    return df
from sklearn.model_selection import train_test_split

def split_data(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

def standard_scale(df):

    df_standard = df.copy()

    scaler = StandardScaler()

    columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    df_standard[columns] = scaler.fit_transform(df_standard[columns])

    return df_standard, scaler

def minmax_scale(df):

    df_minmax = df.copy()

    scaler = MinMaxScaler()

    columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    df_minmax[columns] = scaler.fit_transform(df_minmax[columns])

    return df_minmax, scaler

def correlation_based(df):
    corr_matrix = df.corr(numeric_only=True)
    corr_with_target = corr_matrix['Churn'].abs()
    corr_with_target = corr_with_target.sort_values(ascending=False)
    return corr_with_target

from sklearn.feature_selection import mutual_info_classif
def mutual_information(df):

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    mi_scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI Score': mi_scores
    })

    mi_df = mi_df.sort_values(
        by='MI Score',
        ascending=False
    )

    return mi_df
