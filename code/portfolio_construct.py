# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:07:04 2018

@author: blackmeteor27
"""
import pandas as pd
import datetime 
import os.path
import matplotlib.pyplot as plt
#pick etf by data:
#Date:the date to select ETF(string)
#Factor: the certain feature of dataframe to select ETF
#threshold: certain number of ETF to select
def etf_pick(date,factor, threshold):
    long_stock = []
    short_stock = []
    file = "data/" + date + ".csv"
    data = pd.read_csv(file)
    order_data = data.sort_values(factor,ascending = False)
    long_stock = order_data["ticker"].head(threshold).tolist()
    short_stock = order_data["ticker"].tail(threshold).tolist()
    return long_stock,short_stock

#portfolio return calcultion for each month
#date_str: the beginning date of ETF selection(string)
#long_etf: etf portfolio for long position
#short_etf: etf portfolio for short position
def portfolio_month_return_calculate(date_str,long_etf,short_etf):
    close = pd.read_csv("return/close.csv")
    date = datetime.datetime.strptime(date_str,"%Y-%m-%d")    
    this_month =  date.month   
    this_year = date.year
    while not any(close["Date"] == date_str):
        date = date - datetime.timedelta(1)
        date_str =date.strftime("%Y-%m-%d")
    start = close[close["Date"] == date_str]
    next_month = (this_month + 2) % 12
    next_year = this_year + int((this_month + 2) /12 )
    if next_month == 0:
        next_month = 12
        next_year = next_year - 1
    end_date = datetime.datetime(next_year,next_month,1) - datetime.timedelta(1)
    end_date_str = end_date.strftime("%Y-%m-%d")
    while not any(close["Date"] == end_date_str):
        end_date = end_date - datetime.timedelta(1)
        end_date_str =end_date.strftime("%Y-%m-%d")
    end = close[close["Date"] == end_date_str]
    start_long = start[long_etf]
    start_short = start[short_etf]
    end_long = end[long_etf]
    end_short = end[short_etf]
    return_long= (end_long.sum(1).values - start_long.sum(1).values)/ start_long.sum(1).values   
    return_short = (end_short.sum(1).values - start_short.sum(1).values) / start_short.sum(1).values
    return_total = return_long - return_short
    return return_total
    
#portfolio return calculate between a long period of time and plot each monthly return
#start_date: should be the first day of month, etf selection start from the day before that month
#end_date: should be the first day of month, the return calculate end before that month
#n: the number of ETF to long and short
def portfolio_return_calculate(start_date,end_date,n):
    factor = "quant_composite_technical"
    start = datetime.datetime.strptime(start_date,"%Y-%m-%d")
    end = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    return_month = []
    months_diff = (end.year - start.year)* 12 + end.month - start.month 
    total_return = 1
    for t in range(months_diff):
        month = (start.month + t) % 12
        year = start.year + int((start.month + t) / 12)
        if month == 0:
            month = 12
            year = year - 1 
        date = datetime.date(year,month,1)- datetime.timedelta(1)
        date_str = date.strftime("%Y-%m-%d")
        file = "data/" + date_str + ".csv"
        while not os.path.isfile(file):
            date = date- datetime.timedelta(1)
            date_str = date.strftime("%Y-%m-%d")
            file = "data/" + date_str + ".csv"
        long_etf,short_etf = etf_pick(date_str,factor, n)        
        single_month_return = portfolio_month_return_calculate(date_str,long_etf,short_etf)[0]
        return_month.append(single_month_return)
        total_return *= (1+single_month_return)
    total_return = total_return - 1
    plt.plot(return_month)
    plt.ylabel("monthly return")
    print("total return: %f" %total_return)
    print("monthly return: ")
    print(return_month)
    return total_return

total_return = portfolio_return_calculate("2013-01-01","2013-12-01", 60)
