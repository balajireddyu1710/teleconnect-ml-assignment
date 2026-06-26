# Telco Customer Churn Analysis and Prediction

An end-to-end Machine Learning project for predicting **customer churn** (classification) and **monthly charges** (regression) using the Telco Customer Churn dataset.

---

## Project Overview

This project implements a complete machine learning pipeline for customer analytics using the **Telco Customer Churn** dataset. It consists of two predictive tasks:

* **Customer Churn Prediction (Classification):** Predict whether a customer is likely to discontinue the service.
* **Monthly Charges Prediction (Regression):** Predict the monthly charges based on customer demographics, account information, and subscribed services.

The project covers the complete machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Feature Engineering
* Feature Selection
* Model Training
* Hyperparameter Tuning using GridSearchCV
* Model Evaluation
* Model Interpretation
* Model Serialization using Pickle

---

## Project Structure

```text
teleconnect-ml-assignment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_regression.ipynb
│   └── 05_interpretation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── classifiers.py
│   ├── regressors.py
│   ├── evaluation.py
│   ├── utils.py
│   └── __init__.py
│
├── models/
│   ├── best_classifier.pkl
│   ├── best_regressor.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── classification_report.md
│   ├── regression_report.md
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## Dataset

* **Dataset:** Telco Customer Churn
* **Total Records:** 7,043
* **Classification Target:** Churn
* **Regression Target:** MonthlyCharges

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP
* Pickle
* Jupyter Notebook

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/balajireddyu1710/teleconnect-ml-assignment.git

cd teleconnect-ml-assignment
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Execute the notebooks in the following order:

1. 01_EDA.ipynb
2. 02_preprocessing.ipynb
3. 03_classification.ipynb
4. 04_regression.ipynb
5. 05_interpretation.ipynb

---

## Results

### Classification Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression | **0.8098** | **0.6905** |     0.5160 | **0.5906** | **0.8450** |
| Random Forest       |     0.7890 |     0.6465 |     0.4555 |     0.5344 |     0.8353 |
| Decision Tree       |     0.7512 |     0.5298 | **0.5694** |     0.5489 |     0.7823 |
| KNN                 |     0.7796 |     0.6364 |     0.3986 |     0.4902 |     0.7572 |
| SVM                 |     0.6831 |     0.4262 |     0.5552 |     0.4822 |     0.7254 |

**Best Classification Model:** Logistic Regression

---

### Regression Performance

| Model                          |        MAE |       RMSE |   R² Score |
| ------------------------------ | ---------: | ---------: | ---------: |
| Linear Regression              |     0.7859 |     1.0349 |     0.9988 |
| Ridge Regression               |     0.7859 |     1.0349 |     0.9988 |
| **Lasso Regression**           | **0.7857** | **1.0348** | **0.9988** |
| Elastic Net Regression         |     0.7857 |     1.0348 |     0.9988 |
| Decision Tree Regressor        |     0.8380 |     1.1343 |     0.9985 |
| Random Forest Regressor        | **0.7844** |     1.0722 |     0.9987 |
| Support Vector Regressor (SVR) |     0.8506 |     1.1045 |     0.9986 |

**Best Regression Model:** Lasso Regression

---

## Saved Models

The trained models and preprocessing objects are saved as:

* `best_classifier.pkl`
* `best_regressor.pkl`
* `encoder.pkl`
* `scaler.pkl`

These files can be used for inference without retraining the models.

---

## Author

**Uppaluri Balaji Reddy**

Machine Learning Project – Telco Customer Churn Analysis and Prediction
