# 📊 Machine Learning Data Preprocessing & Modeling Tool

An interactive, end-to-end web application built with **Streamlit** designed to automate and streamline the entire data science pipeline. This tool provides a user-friendly dashboard to upload raw datasets, perform comprehensive data cleaning, handle missing values structurally, visualize relationships, handle class imbalances, and train/evaluate standard Machine Learning models.

---

## 🚀 Key Modules & Features

### 1. 📁 Data Upload & Management
* Supports seamless ingestion of **CSV** and **Excel** (`.xlsx`) datasets.
* Tracks data mutations dynamically with custom reset functions to restore original data shapes.

### 2. 📊 Interactive Data Visualization
* Generates continuous insight plots using **Scatter Plots**, **Line Plots**, and **Box Plots** built natively with `seaborn` and `matplotlib`.

### 3. 🔧 Structural Missing Values Imputation
* Implements multi-strategy missing value treatments:
  * **Simple Imputer:** Mean, Median, and Most Frequent strategies.
  * **KNN Imputer:** Distance-based numeric neighborhood imputation ($k$-neighbors tuning).
  * **Iterative Imputer:** Multivariate chain equations for statistical reconstruction.
* Built-in dynamic data type protection to ensure categorical data is encoded before execution of distance algorithms.

### 4. 🏷️ Categorical Data Encoding
* **Label Encoding:** Maps labels to numerical values while carefully preserving native missing flags (`NaN`) to prevent statistical leakages.
* **One-Hot Encoding:** Generates binary dummy variables with individual `NaN` indicators.

### 5. 📏 Scaling, Normalization & Transformations
* Scales data values safely via **Min-Max Scaling** and **Standard Scaling**.
* Supports feature extraction via structural feature engineering and mathematical transformations.

### 6. 📦 Outlier Analysis & Treatment
* Filters or clips anomalous records utilizing statistical boundary rules:
  * **IQR Method:** Interquartile range outlier deletion.
  * **Z-Score Method:** Absolute deviation boundary clipping ($|Z| \le 3$).
  * **Winsorization:** Limits extreme value distributions via systematic quantiles using `scipy`.

### 7. ⚖️ Handling Class Imbalances
* Balances target variables using resampling techniques:
  * **Oversampling:** Synthetic Minority Over-sampling Technique (**SMOTE**).
  * **Undersampling:** Random Under Sampler and **Cluster Centroids**.

### 8. 🎯 Feature Selection & Extraction
* Identifies optimal analytical representations using **Recursive Feature Elimination (RFE)** and **Principal Component Analysis (PCA)**.

### 9. 🦾 Predictive Modeling & Evaluation
* Trains and tests baseline supervised and unsupervised models: **Logistic Regression, SVM, Random Forest, Decision Trees, KNN, Naive Bayes, MLP Neural Networks**, and **K-Means Clustering**.
* Generates interactive diagnostic reporting: Accuracy, Precision, Recall, F1-Score, ROC-AUC curves, MAE, MSE, $R^2$, and Silhouette Scores.

---

## 🛠️ Technology Stack

* **Dashboard Framework:** Streamlit
* **Data Core:** Pandas, NumPy
* **Scientific Computation:** SciPy
* **Visualization Layer:** Seaborn, Matplotlib
* **Machine Learning Engine:** Scikit-Learn, Imbalanced-Learn

---

## 💻 Setup & Installation Guide

1. **Clone this repository to your local directory:**
   ```bash
   git clone [https://github.com/nourayman7506data/Machine-Learning-College-Project.git](https://github.com/nourayman7506data/Machine-Learning-College-Project.git)
   cd Machine-Learning-College-Project
