



1\. Project Overview



This report summarizes the performance of multiple supervised machine learning regression models developed to predict customer \*\*MonthlyCharges\*\* using the Telco Customer Churn dataset. The objective is to accurately estimate the monthly charges of customers based on demographic, account, and service-related attributes, enabling better analysis of customer billing patterns.







2\. Dataset Information



Dataset: Telco Customer Churn

Problem Type: Regression

Target Variable: MonthlyCharges

Total Records:7,043

Train/Test Split: 70% Training, 15% Validation, 15% Testing



\-



&#x20;3. Data Preprocessing



The following preprocessing techniques were applied before model training:



\* Missing value handling

\* Label Encoding of binary categorical variables

\* One-Hot Encoding of categorical features

\* Feature Engineering



&#x20; \* Average Monthly Spend

&#x20; \* Service Count

&#x20; \* Contract Value

\* Standard Scaling for Lasso Regression and Support Vector Regressor (SVR)

\* Stratified Train/Validation/Test Split

\* Hyperparameter tuning using GridSearchCV

\* Model serialization using Pickle







4\. Regression Models Evaluated



The following supervised machine learning regression algorithms were trained and evaluated:



\* Linear Regression

\* Ridge Regression

\* Lasso Regression

\* Elastic Net Regression

\* Decision Tree Regressor

\* Random Forest Regressor

\* Support Vector Regressor (SVR)









5\. Evaluation Metrics



The following metrics were used to compare model performance:



\* Mean Absolute Error (MAE)

\* Mean Squared Error (MSE)

\* Root Mean Squared Error (RMSE)

\* Coefficient of Determination (R² Score)

\* Adjusted R² Score

\* Training Time



\---



&#x20;6. Performance Comparison



| Model                          |      MAE |      MSE |     RMSE | R² Score | Adjusted R² | Training Time (s) |

| ------------------------------ | -------: | -------: | -------: | -------: | ----------: | ----------------: |

| Linear Regression              | 0.785868 | 1.071030 | 1.034906 | 0.998793 |    0.998757 |          0.067227 |

| Ridge Regression               | 0.785850 | 1.070991 | 1.034887 | 0.998793 |    0.998757 |          5.032951 |

| Lasso Regression               | 0.785692 | 1.070839 | 1.034813 | 0.998793 |    0.998757 |          0.124784 |

| Elastic Net Regression         | 0.785677 | 1.070902 | 1.034844 | 0.998793 |    0.998757 |          2.383212 |

| Decision Tree Regressor        | 0.837972 | 1.286676 | 1.134317 | 0.998549 |    0.998507 |          1.176767 |

| Random Forest Regressor        | 0.784375 | 1.149523 | 1.072158 | 0.998704 |    0.998666 |         18.165512 |

| Support Vector Regressor (SVR) | 0.850633 | 1.220002 | 1.104537 | 0.998625 |    0.998584 |         20.862295 |



\---



&#x20;7. Best Performing Model



Lasso Regression



Lasso Regression achieved the best overall performance among all evaluated regression models.



&#x20;Reasons



\* Lowest Mean Squared Error (MSE = 1.070839)

\* Lowest Root Mean Squared Error (RMSE = 1.034813)

\* Highest R² Score (0.998793)

\* Highest Adjusted R² Score (0.998757)

\* Fast training time (0.124784 seconds)

\* L1 regularization helps reduce overfitting and produces a simpler, more interpretable model by shrinking less important feature coefficients toward zero.



Although Random Forest Regressor achieved the lowest MAE,Lasso Regression provided the best overall balance between prediction accuracy, computational efficiency, and model generalization.







&#x20;8. Model Interpretation



To improve model interpretability, multiple explanation techniques were applied.



Linear Models



\* Regression coefficients were analyzed for Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net Regression to understand the direction and magnitude of each feature's influence on monthly charges.



Tree-Based Models



\* Feature importance scores were extracted from Decision Tree and Random Forest Regressors to identify the most influential predictors affecting customer monthly charges.



\---



9\. Visualizations Generated



The following visualizations were produced during model evaluation:



\* Actual vs Predicted Scatter Plot

\* Residual Error Distribution

\* Regression Coefficient Importance Plot

\* Tree-Based Feature Importance Plot



All figures are available in the reports/figures/\*\* directory.







10\. Conclusion



Seven supervised machine learning regression algorithms were evaluated to predict customer MonthlyCharges.



Among the evaluated models, Lasso Regression demonstrated the best overall predictive performance by achieving the lowest MSE and RMSE while maintaining an excellent R² Score of 0.998793. Linear Regression, Ridge Regression, and Elastic Net Regression produced nearly identical results, indicating that the relationship between the input features and monthly charges is predominantly linear. Although Random Forest Regressor achieved the lowest MAE, it required substantially more training time. Decision Tree Regressor and Support Vector Regressor (SVR) showed comparatively higher prediction errors.



The developed regression pipeline includes preprocessing, feature engineering, hyperparameter optimization, model evaluation, model interpretation, visualization, and model persistence, providing a complete and reproducible machine learning workflow.



\---



\## 11. Future Work



Future improvements may include:



\* Gradient Boosting methods (XGBoost, LightGBM, CatBoost)

\* Bayesian Hyperparameter Optimization

\* Advanced feature selection techniques

\* Ensemble regression methods

\* Model deployment using Flask or FastAPI

\* Real-time Monthly Charges prediction dashboard



