# -*- coding: utf-8 -*-
"""
JPMorgan Chase & Co. Quantitative Research Virtual Experience Program
Task 3 - Model Answer

Credit Risk Analysis
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

# import loan book data
loan_book = pd.read_csv(r"Task 3 and 4_Loan_Data.csv")
loan_book.info()

loan_book.value_counts('default') # highlights imbalance of outcome, approx 4.5:1

# split into features and outcome for training and testing
X = loan_book.drop(['customer_id', 'default'], axis=1)
y = loan_book['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.66)

# instantiate logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_test, y_test)

pred = model.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, pred)

#optimise logistic regression model
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

grid = dict(solver=solvers, penalty=penalty , C=c_values)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
new_model = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc', error_score=0)
grid_result = new_model.best_estimator_.fit(X_train, y_train)

grid_pred = new_model.best_estimator_.predict_proba(X_test)[:,1]
new_roc = roc_auc_score(y_test, grid_pred)

# quick estimate of feature importance
np.std(X, 0).values*new_model.best_estimator_.coef_

# function to return probability of default and expected loss
def default_function(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, 
                     income, years_employed, fico_score):
    
    X = np.array([credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, 
                 income, years_employed, fico_score]).reshape(1,-1)
    
    pd = new_model.predict_proba(X)[:,1]

    return pd, pd*0.9*total_debt_outstanding # EL = PD × (1 − RR) × EAD

credit_lines_outstanding = 0.000000
loan_amt_outstanding = 5221.545193
total_debt_outstanding = 3915.471226
income = 28039.385460
years_employed = 0.5000000
fico_score = 605.000000

prob_def, exp_loss = default_function(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, 
                                      income, years_employed, fico_score)

print(f"Probability of Deafult = {prob_def[0]}, Expected Loss = {exp_loss[0]}")
