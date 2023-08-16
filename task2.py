# -*- coding: utf-8 -*-
"""
JPMorgan Chase & Co. Quantitative Research Virtual Experience Program
Task 2

Price a commodity storage contract
"""
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

# read in daily price data created in task 1 - convert date index to datetime object
data = pd.read_csv('daily_data.csv', index_col=0, header=0, parse_dates=['Dates'])

def pricing(inj_dates: list, with_dates: list, 
            prices: pd.DataFrame, rate: float, 
            max_vol:float, monthly_store_cost: float):
    
    """
    A function that takes injection, wihtdrawal and storage information to retrun
    the price of a commodity storage contract.
    """
    
    # converting list of string dates to datetime objects
    inj_dates = [dt.datetime.strptime(date, '%d/%m/%Y') for date in inj_dates]
    with_dates = [dt.datetime.strptime(date, '%d/%m/%Y') for date in with_dates]
    
    # caluclate the number of days between injection and withdrawal to find storage costs
    dates_diff = [int(relativedelta(x, y).months) for x, y in zip(with_dates, inj_dates)]
    total_store_costs = monthly_store_cost*sum(dates_diff)
    
    # calculate price differences between injection and withdrawal
    inj_prices = np.array([prices.loc[date, 'Prices'] for date in inj_dates])
    with_prices = np.array([prices.loc[date, 'Prices'] for date in with_dates])
    prices_diff = sum(with_prices - inj_prices)
    
    # determine cost to inject/withdraw (assuming max volume injected/withdrawn)
    inj_costs = len(inj_dates)*max_vol*rate
    with_costs = len(with_dates)*max_vol*rate
        
    return prices_diff - total_store_costs - (inj_costs + with_costs) 

print(pricing(['01/01/2022'], ['01/03/2022'], data, 0.01, 5, 0.1))