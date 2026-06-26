

1\. Project Overview



This report summarizes the performance of multiple regression algorithms developed to predict \*\*MonthlyCharges\*\* using the Telco Customer Churn dataset. The objective is to accurately estimate the monthly charges of customers based on demographic, account, and service-related attributes.



\---



2\. Dataset Information



| \*\*Attribute\*\*       | \*\*Value\*\*            |

| ------------------- | -------------------- |

| \*\*Dataset\*\*         | Telco Customer Churn |

| \*\*Problem Type\*\*    | Regression           |

| \*\*Target Variable\*\* | MonthlyCharges       |

| \*\*Total Records\*\*   | 7,043                |

| \*\*Training Set\*\*    | 70% (4,930 records)  |

| \*\*Validation Set\*\*  | 15% (1,056 records)  |

| \*\*Testing Set\*\*     | 15% (1,057 records)  |





3\. Data Preprocessing



The following preprocessing techniques were applied before model training:



\- Missing value handling

\- Label Encoding of binary categorical features

\- One-Hot Encoding of categorical variables

\- Feature Engineering

&#x20; - Average Monthly Spend

&#x20; - Service Count

&#x20; - Contract Value

\- Standard Scaling for Lasso and SVR

\- Hyperparameter tuning using GridSearchCV

\- Model persistence using Pickle





4\. Regression Models Evaluated



The following regression algorithms were trained and evaluated:



\- Linear Regression

\- Ridge Regression

\- Lasso Regression

\- Elastic Net Regression

\- Decision Tree Regressor

\- Random Forest Regressor

\- Support Vector Regressor (SVR)





5\. Evaluation Metrics



The models were evaluated using the following metrics:



\- Mean Absolute Error (MAE)

\- Mean Squared Error (MSE)

\- Root Mean Squared Error (RMSE)

\- Coefficient of Determination (R²)

\- Adjusted R²

\- Training Time



\---



6\. Model Performance Comparison



| Model | MAE | MSE | RMSE | R² | Adjusted R² | Training Time (s) |

|-------|-----|-----|------|----|-------------|-------------------|

| Linear Regression | 0.785868 | 1.071030 | 1.034906 | 0.998793 | 0.998757 | 0.067227 |

| Ridge Regression | 0.785850 | 1.070991 | 1.034887 | 0.998793 | 0.998757 | 5.032951 |

| Lasso Regression | 0.785692 | 1.070839 | 1.034813 | 0.998793 | 0.998757 | 0.124784 |

| Elastic Net Regression | 0.785677 | 1.070902 | 1.034844 | 0.998793 | 0.998757 | 2.383212 |

| Decision Tree Regressor | 0.837972 | 1.286676 | 1.134317 | 0.998549 | 0.998507 | 1.176767 |

| Random Forest Regressor | 0.784375 | 1.149523 | 1.072158 | 0.998704 | 0.998666 | 18.165512 |

| Support Vector Regressor (SVR) | 0.850633 | 1.220002 | 1.104537 | 0.998625 | 0.998584 | 20.862295 |          



\---

7\. Best Performing Model



Lasso Regression



Lasso Regression achieved the best overall performance among the evaluated regression models.



Reasons



\- Lowest Mean Squared Error (\*\*MSE = 1.070839\*\*)

\- Lowest Root Mean Squared Error (\*\*RMSE = 1.034813\*\*)

\- Highest R² Score (\*\*0.998793\*\*)

\- Highest Adjusted R² (\*\*0.998757\*\*)

\- Fast training time (\*\*0.124784 seconds\*\*)

\- Produces a simple and interpretable model through L1 regularization.



Although Random Forest achieved the lowest MAE, Lasso Regression demonstrated the best balance between prediction accuracy, computational efficiency, and model generalization.



\---

8\. Model Interpretation



The regression models were analyzed to understand the contribution of input features.



\### Linear Models



\- Feature coefficients were examined for Linear Regression, Ridge, Lasso, and Elastic Net to identify how each feature influenced the predicted monthly charges.



\### Tree-Based Models



\- Tree-based feature importance analysis was performed for Decision Tree and Random Forest Regressors to determine the most influential predictors.







9\. Visualizations Generated



The following visualizations were produced during regression model evaluation:



\- Actual vs Predicted Scatter Plot

\- Residual Error Distribution

\- Coefficient Importance Plot

\- Tree-based Feature Importance Plot



All figures are available in the \*\*reports/figures/\*\* directory.





\---

10\. Conclusion



Seven regression algorithms were evaluated to predict customer \*\*MonthlyCharges\*\* using the Telco Customer Churn dataset.



Among the evaluated models, \*\*Lasso Regression\*\* demonstrated the best overall predictive performance by achieving the lowest MSE and RMSE while maintaining an excellent R² score of \*\*0.998793\*\*. The model effectively applies L1 regularization, improving generalization while maintaining high prediction accuracy. Linear Regression, Ridge Regression, and Elastic Net produced nearly identical results, indicating a strong linear relationship between the input features and the target variable. Random Forest achieved the lowest MAE but required substantially longer training time, whereas Decision Tree and SVR exhibited comparatively higher prediction errors.



Overall, the regression pipeline incorporates preprocessing, feature engineering, hyperparameter optimization, evaluation, visualization, model interpretation, and model persistence, resulting in a complete, reproducible, and efficient machine learning workflow.



\---



11\. Future Work



Possible improvements include:



\- Gradient Boosting Regressors (XGBoost, LightGBM, CatBoost)

\- Bayesian Hyperparameter Optimization

\- Automated Feature Selection

\- Ensemble Regression Models

\- Model Deployment using Flask or FastAPI

\- Real-time Monthly Charges Prediction Dashboard

