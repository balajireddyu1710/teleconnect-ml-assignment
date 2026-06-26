## Dataset

### Dataset Overview

The project uses the **Telco Customer Churn** dataset, which contains customer demographic information, account details, subscribed services, billing information, and churn status. The dataset is widely used for customer churn prediction and machine learning research.

| Attribute                 | Value                           |
| ------------------------- | ------------------------------- |
| **Dataset Name**          | Telco Customer Churn            |
| **Source**                | IBM Sample Dataset (Kaggle)     |
| **Problem Type**          | Classification and Regression   |
| **Classification Target** | Churn                           |
| **Regression Target**     | MonthlyCharges                  |
| **Total Records**         | 7,043                           |
| **Total Features**        | 21 (including target variables) |

---

### Dataset Source

The dataset is publicly available on Kaggle and can be downloaded from:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

### Data Dictionary

| Feature              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **customerID**       | Unique identifier assigned to each customer.                                                |
| **gender**           | Gender of the customer (Male/Female).                                                       |
| **SeniorCitizen**    | Indicates whether the customer is a senior citizen (0 = No, 1 = Yes).                       |
| **Partner**          | Indicates whether the customer has a partner.                                               |
| **Dependents**       | Indicates whether the customer has dependents.                                              |
| **tenure**           | Number of months the customer has remained with the company.                                |
| **PhoneService**     | Indicates whether the customer subscribes to phone service.                                 |
| **MultipleLines**    | Indicates whether the customer has multiple phone lines.                                    |
| **InternetService**  | Type of internet service subscribed (DSL, Fiber Optic, or No Internet).                     |
| **OnlineSecurity**   | Indicates whether online security service is subscribed.                                    |
| **OnlineBackup**     | Indicates whether online backup service is subscribed.                                      |
| **DeviceProtection** | Indicates whether device protection service is subscribed.                                  |
| **TechSupport**      | Indicates whether technical support service is subscribed.                                  |
| **StreamingTV**      | Indicates whether streaming TV service is subscribed.                                       |
| **StreamingMovies**  | Indicates whether streaming movie service is subscribed.                                    |
| **Contract**         | Type of customer contract (Month-to-Month, One Year, or Two Year).                          |
| **PaperlessBilling** | Indicates whether paperless billing is enabled.                                             |
| **PaymentMethod**    | Payment method used by the customer.                                                        |
| **MonthlyCharges**   | Monthly amount charged to the customer. *(Regression Target)*                               |
| **TotalCharges**     | Total charges accumulated during the customer's tenure.                                     |
| **Churn**            | Indicates whether the customer discontinued the service (Yes/No). *(Classification Target)* |

---

### Dataset Usage

The dataset was used for two supervised machine learning tasks:

* **Classification:** Predict whether a customer will churn based on demographic, service, and billing information.
* **Regression:** Predict the customer's monthly charges using demographic and service-related features.

Before model training, the dataset underwent preprocessing, including missing value handling, categorical encoding, feature engineering, feature scaling, feature selection, and train-validation-test splitting.
