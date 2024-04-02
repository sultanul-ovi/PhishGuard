# Written by Ovi
# October 24, 2023
# Load the phishing dataset into a Pandas DataFrame

import pandas as pd

df = pd.read_csv('dataset.csv')

# Written by Ovi
# October 24, 2023
# Feature Engineering: Select relevant features for the analysis

selected_features = df[['having_IPhaving_IP_Address', 'URLURL_Length', 'SSLfinal_State', 'web_traffic', 'Page_Rank']]

# Written by Ovi
# October 24, 2023
# Training individual machine learning models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = selected_features
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# SVM
svm = SVC()
svm.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# Written by Ovi
# October 24, 2023
# Implementing an ensemble method using majority voting

from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm', svm), ('dt', dt), ('knn', knn)], voting='hard')
ensemble.fit(X_train, y_train)

# Written by Ovi
# October 24, 2023
# Evaluation using Point Estimates and Confidence Intervals

from sklearn.metrics import accuracy_score

y_pred_ensemble = ensemble.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_knn = knn.predict(X_test)

print(f'Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble)}')
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}')
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}')
print(f'Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}')
print(f'K-Nearest Neighbors Accuracy: {accuracy_score(y_test, y_pred_knn)}')

# Written by Ovi
# October 24, 2023
# Complete Python script to evaluate and perform statistical analysis on ensemble and individual models

import pandas as pd
import numpy as np
from scipy.stats import sem, t, ttest_rel
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm
accs_ensemble = [0.9, 0.91, 0.92, 0.89, 0.93] # need to fix
# Function to calculate confidence intervals
def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    interval = std_err * t.ppf((1 + confidence) / 2., n - 1)
    return m, m - interval, m + interval

# Load your dataset here
# df = pd.read_csv('your_dataset.csv')

# Data Preparation, Feature Engineering, and Model Training
# ... (insert your code here)

# Perform K-Fold Cross-validation to obtain lists of accuracy scores
# Example lists: accs_ensemble, accs_lr, accs_rf, etc.

# Point Estimates and Confidence Intervals
point_estimate, lower_bound, upper_bound = confidence_interval(accs_ensemble)
print(f'Ensemble Model: Mean Accuracy = {point_estimate}, Confidence Interval = ({lower_bound}, {upper_bound})')

# Comparing Proportions using Z-test
stat, pval = proportions_ztest([successes_ensemble, successes_lr], [n_ensemble, n_lr])

# Paired/Unpaired Observations using t-test
t_stat, p_val = ttest_rel(accs_ensemble, accs_lr)

# Hypothesis Testing
if p_val < 0.05:
    print("Ensemble model is significantly better")
else:
    print("No significant difference between ensemble and individual model")

# Regression Analysis
# Create a DataFrame with individual models' accuracy as features
# and the ensemble model's accuracy as the target variable
# Example: df_regression = pd.DataFrame({'lr': accs_lr, 'rf': accs_rf, 'ensemble': accs_ensemble})
X = df_regression[['lr', 'rf']]  # include all individual models here
y = df_regression['ensemble']

X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X)
result = model.fit()
print(result.summary())


# Support Vector Machines (SVM): Particularly good for high-dimensional data.
# Random Forest: An ensemble method itself, can capture complex decision boundaries.
# Gradient Boosting: Similar to Random Forest but often provides better performance.
# K-Nearest Neighbors (KNN): Simple but effective for certain types of data.
# Naive Bayes: Particularly useful for text-based features if you have any.
# Neural Networks: A deep learning model could capture non-linear relationships.
# Decision Trees: Simple to implement and interpret, can serve as a baseline model.
# Logistic Regression: Though simple, it can be very effective when feature engineering is done right.
# AdaBoost: Another ensemble technique that adjusts the weights of weak classifiers.
# CatBoost/LightGBM/XGBoost: Advanced boosting algorithms

# how about these following steps: how to include them in our project? 
# Evaluation and Statistical Analysis
# Evaluate the model using Point Estimates, Confidence Intervals, and Hypothesis Testing.
# Use Comparing Proportions and Paired/Unpaired Observations to compare the ensemble model with individual models.
# Apply Regression Analysis to measure the contribution of individual models in the ensemble.
# Statistical Techniques:
# Point Estimates and Confidence Intervals: For model accuracy and feature importance.
# Comparing Proportions: To compare the success rates of different models.
# Paired/Unpaired Observations: For comparing ensemble models vs. single models.
# Hypothesis Testing: To rigorously evaluate whether the ensemble model significantly outperforms individual models.
# Regression Analysis: To understand the impact of individual models on the ensemble's performance.