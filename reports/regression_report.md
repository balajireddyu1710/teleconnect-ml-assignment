# Regression Model Evaluation Report

## 1. Project Overview

This report summarizes the performance of multiple supervised machine learning regression models developed to predict customer **MonthlyCharges** using the Telco Customer Churn dataset. The objective is to accurately estimate the monthly charges of customers based on demographic, account, and service-related attributes, enabling better analysis of customer billing behavior and pricing patterns.

---

## 2. Dataset Information

**Dataset:** Telco Customer Churn

**Problem Type:** Regression

**Target Variable:** MonthlyCharges

**Total Records:** 7,043

**Train/Test Split:** 70% Training, 15% Validation, 15% Testing

---

## 3. Data Preprocessing

The following preprocessing techniques were applied before model training:

* Missing value handling
* Label Encoding of binary categorical variables
* One-Hot Encoding of categorical features
* Feature Engineering

  * Average Monthly Spend
  * Service Count
  * Contract Value
* Standard Scaling for Lasso Regression and Support Vector Regressor (SVR)
* Train/Validation/Test Split
* Hyperparameter tuning using GridSearchCV
* Model serialization using Pickle

---

## 4. Regression Models Evaluated

The following supervised machine learning algorithms were trained and evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Elastic Net Regression
* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (SVR)

---

## 5. Evaluation Metrics

The following metrics were used to compare model performance:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R² Score)
* Adjusted R² Score
* Training Time

---

## 6. Performance Comparison

| Model                          |      MAE |      MSE |     RMSE | R² Score | Adjusted R² | Training Time (s) |
| ------------------------------ | -------: | -------: | -------: | -------: | ----------: | ----------------: |
| Linear Regression              | 0.785868 | 1.071030 | 1.034906 | 0.998793 |    0.998757 |          0.067227 |
| Ridge Regression               | 0.785850 | 1.070991 | 1.034887 | 0.998793 |    0.998757 |          5.032951 |
| Lasso Regression               | 0.785692 | 1.070839 | 1.034813 | 0.998793 |    0.998757 |          0.124784 |
| Elastic Net Regression         | 0.785677 | 1.070902 | 1.034844 | 0.998793 |    0.998757 |          2.383212 |
| Decision Tree Regressor        | 0.837972 | 1.286676 | 1.134317 | 0.998549 |    0.998507 |          1.176767 |
| Random Forest Regressor        | 0.784375 | 1.149523 | 1.072158 | 0.998704 |    0.998666 |         18.165512 |
| Support Vector Regressor (SVR) | 0.850633 | 1.220002 | 1.104537 | 0.998625 |    0.998584 |         20.862295 |

---

## 7. Best Performing Model

### Lasso Regression

Lasso Regression achieved the best overall performance among all evaluated regression models.

### Reasons

* Lowest Mean Squared Error (**1.070839**)
* Lowest Root Mean Squared Error (**1.034813**)
* Highest R² Score (**0.998793**)
* Highest Adjusted R² Score (**0.998757**)
* Very fast training time (**0.124784 seconds**)
* L1 regularization improves model generalization while reducing unnecessary feature coefficients, making the model more interpretable.

Although **Random Forest Regressor** achieved the lowest Mean Absolute Error (MAE), **Lasso Regression** demonstrated the best balance between prediction accuracy, computational efficiency, and model simplicity.

---

## 8. Model Interpretation

To improve model interpretability, multiple explanation techniques were applied.

### Linear Regression Models

* Regression coefficients were analyzed for Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net Regression to determine the direction and magnitude of each feature's contribution toward predicting MonthlyCharges.

### Decision Tree Regressor

* Feature importance scores were extracted to identify the most influential predictors used during tree construction.

### Random Forest Regressor

* Ensemble feature importance analysis was performed to determine which variables contributed most significantly to MonthlyCharges prediction.

### Coefficient Comparison

* The coefficients of Linear Regression, Ridge Regression, and Lasso Regression were compared to observe the effect of regularization on feature weights. Lasso Regression reduced less important coefficients toward zero, producing a simpler and more interpretable model.

---

## 9. Visualizations Generated

The following visualizations were produced during model evaluation:

* Actual vs Predicted Scatter Plot
* Residual Error Distribution
* Linear Regression Coefficient Importance
* Ridge Regression Coefficient Importance
* Lasso Regression Coefficient Importance
* Decision Tree Feature Importance
* Random Forest Feature Importance
* Regression Model Performance Comparison

All figures are available in the **reports/figures/** directory.

---

## 10. Conclusion

Seven supervised machine learning regression algorithms were evaluated for predicting customer **MonthlyCharges**.

Among the evaluated models, **Lasso Regression** demonstrated the best overall predictive performance by achieving the lowest Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) while maintaining an excellent R² Score of **0.998793**. Linear Regression, Ridge Regression, and Elastic Net Regression produced nearly identical results, indicating a strong linear relationship between the selected features and the target variable. Although Random Forest Regressor achieved the lowest Mean Absolute Error (MAE), it required substantially longer training time. Decision Tree Regressor and Support Vector Regressor (SVR) exhibited comparatively higher prediction errors.

The developed regression pipeline includes preprocessing, feature engineering, hyperparameter optimization, model evaluation, visualization, model interpretation, and model persistence, providing a complete, reproducible, and efficient machine learning workflow.

---

## 11. Future Work

Future improvements may include:

* Gradient Boosting methods (XGBoost, LightGBM, CatBoost)
* Bayesian Hyperparameter Optimization
* Advanced feature selection techniques
* Ensemble regression models
* Automated Machine Learning (AutoML)
* Model deployment using Flask or FastAPI
* Real-time MonthlyCharges prediction dashboard
