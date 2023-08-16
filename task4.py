# -*- coding: utf-8 -*-
"""
JPMorgan Chase & Co. Quantitative Research Virtual Experience Program
Task 4

Bucket FICO scores
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

loan_book = pd.read_csv(r"Task 3 and 4_Loan_Data.csv")
fico_scores = loan_book['fico_score']
defaults = loan_book['default']

# Define the number of bins
num_bins = 10
    
# Define negative log-likelihood function
def neg_log_likelihood(fico_bins):
    
    neg_ll = []
    for i in range(num_bins):
        mask = (fico_scores >= fico_bins[i]) & (fico_scores < fico_bins[i + 1])
        k = np.sum(defaults[mask])
        n = len(defaults[mask])
        p = k/n
        if (p==0 or p==1):
            neg_ll.append(0)
        neg_ll_i = -np.sum(k*np.log(p)+(n-k)*np.log(1-p))
        neg_ll.append(neg_ll_i)
            
    return -np.sum(neg_ll)

# Initial parameter guesses for optimisation
percentiles = np.linspace(0, 100, num_bins + 1)
initial_params = np.percentile(fico_scores, percentiles)

# Optimize the negative log-likelihood
result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead')

# Get optimal bins
optimal_bins = result.x
optimal_bins[0] = 300
optimal_bins[-1] = 850

print(optimal_bins)

# Bin data based on FICO scores
labels = np.arange(num_bins, 0, -1)
loan_book["fico_bucket"] = pd.cut(loan_book['fico_score'], optimal_bins, include_lowest=True, labels=labels)

loan_book.groupby("fico_bucket")['default'].mean()

