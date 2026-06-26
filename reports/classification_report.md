

1. Project Overview

This report summarizes the performance of multiple supervised machine learning classification models developed to predict customer churn using the Telco Customer Churn dataset. The objective is to accurately identify customers who are likely to discontinue the service, enabling businesses to implement effective customer retention strategies.

---

2. Dataset Information

Dataset: Telco Customer Churn
Problem Type: Binary Classification
Target Variable: Churn
Total Records: 7,043
Train/Test Split: 70% Training, 15% Validation, 15% Testing


3. Data Preprocessing

The following preprocessing techniques were applied before model training:

* Missing value handling
* Label Encoding of binary categorical variables
* One-Hot Encoding of categorical features
* Feature Engineering
  * Average Monthly Spend
  * Service Count
  * Contract Value
* Stratified Train/Validation/Test Split
* Hyperparameter tuning using GridSearchCV
* Model serialization using Pickle


4. Classification Models Evaluated

The following machine learning algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)



5. Evaluation Metrics

The following metrics were used to compare model performance:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Training Time

---

6. Performance Comparison

|   Model             |   Accuracy   |   Precision   |   Recall   |   F1 Score   |   ROC-AUC   |   Training Time (s)   |
| ------------------- | -----------: | ------------: | ---------: | -----------: | ----------: | --------------------: |
| Logistic Regression |     0.809839 |      0.690476 |   0.516014 |     0.590631 |    0.845035 |              5.020673 |
| Random Forest       |     0.789026 |      0.646465 |   0.455516 |     0.534447 |    0.835315 |              4.846838 |
| Decision Tree       |     0.751183 |      0.529801 |   0.569395 |     0.548885 |    0.782311 |              0.684566 |
| KNN                 |     0.779565 |      0.636364 |   0.398577 |     0.490153 |    0.757225 |              0.473327 |
| SVM                 |     0.683065 |      0.426230 |   0.555160 |     0.482226 |    0.725385 |              3.705269 |

7. Best Performing Model

Logistic Regression

Logistic Regression achieved the best overall performance among all evaluated classifiers.

Reasons

* Highest Accuracy (**80.98%**)
* Highest Precision (**69.05%**)
* Highest F1 Score (**59.06%**)
* Highest ROC-AUC (**0.845**)
* Good balance between identifying churn customers and minimizing false predictions.

Although the Decision Tree achieved the highest Recall, its lower Precision and Accuracy resulted in a less balanced overall performance.



8. Model Interpretation

To improve model interpretability, multiple explanation techniques were applied.

 Logistic Regression

* Feature coefficients were analyzed to determine the direction and magnitude of each feature's contribution to churn prediction.

 Random Forest

* Feature importance scores were extracted to identify the most influential variables affecting customer churn.

 SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to provide both global and local explanations of model predictions, helping understand the impact of individual features on customer churn.



9. Visualizations Generated

The following visualizations were produced during model evaluation:

* Class Distribution
* Correlation Heatmap
* Confusion Matrix for each classifier
* ROC Curve for each classifier
* Logistic Regression Coefficient Importance
* Random Forest Feature Importance
* SHAP Summary Plot

All figures are available in the **reports/figures/** directory.



 10. Conclusion

Five supervised machine learning classification algorithms were evaluated for customer churn prediction.

Among the evaluated models, **Logistic Regression** demonstrated the best overall performance by achieving the highest Accuracy, Precision, F1 Score, and ROC-AUC score. While Decision Tree provided higher Recall, Logistic Regression maintained a better balance across all evaluation metrics, making it the most suitable model for this customer churn prediction task.

The developed classification pipeline includes preprocessing, feature engineering, hyperparameter optimization, model evaluation, model interpretation, and model persistence, providing a complete and reproducible machine learning workflow.

---

 11. Future Work

Future improvements may include:

* Gradient Boosting methods (XGBoost, LightGBM, CatBoost)
* Advanced feature selection techniques
* Probability calibration for improved confidence estimates
* Automated machine learning (AutoML)
* Model deployment using Flask or FastAPI
* Real-time customer churn prediction dashboard
