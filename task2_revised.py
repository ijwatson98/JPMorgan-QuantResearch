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
            max_storage_vol: float, monthly_storage_cost: float,
            injection_withdrawal_cost: float):
    
    """
    A function that takes injection, wihtdrawal and storage information to retrun
    the price of a commodity storage contract.
    """
    
    curr_vol = 0
    costs = 0
    income = 0
    
    # converting list of string dates to datetime objects
    inj_dates = [dt.datetime.strptime(date, '%d/%m/%Y') for date in inj_dates]
    with_dates = [dt.datetime.strptime(date, '%d/%m/%Y') for date in with_dates]
    dates = sorted(set(inj_dates+with_dates))
    
    for date in dates:
        
        if date in inj_dates:
            if curr_vol + rate <= max_storage_vol:
                curr_vol += rate
                costs +=  prices.loc[date, 'Prices']*rate
                inj_cost = injection_withdrawal_cost*rate
                costs += inj_cost
                print(f"Injected gas on {date} at a price of {prices.loc[date, 'Prices']}")
            else:
                print("Not enough storage space to inject")
                
        if date in with_dates:
            if rate <= curr_vol:
                curr_vol -= rate
                income +=  prices.loc[date, 'Prices']*rate
                with_cost = injection_withdrawal_cost*rate
                income -= with_cost
                print(f"Withdrew gas on {date} at a price of {prices.loc[date, 'Prices']}")
            else:
                print("Not enough stored to withdraw")
                
            storage_period = relativedelta(max(with_dates) - min(inj_dates)).months
            costs += storage_period*monthly_storage_cost
        
    return income - costs

inj_dates = ["01/01/2022", "01/02/2022"]
with_dates = in_dates = ["27/01/2022", "15/02/2022"]
rate = 100000  # rate of gas that can be injected/withdrawn in cubic feet per day
monthly_storage_cost = 10000  # total volume in cubic feet
injection_withdrawal_cost = 0.0005  # $/cf
max_storage_vol = 500000 # maximum storage capacity of the storage facility
 
print(pricing(inj_dates, with_dates, data, rate, max_storage_vol, monthly_storage_cost,
              injection_withdrawal_cost))