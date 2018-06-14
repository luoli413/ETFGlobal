# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import glob
import sklearn.linear_model as lm
import os.path

#1 Data preparation
#1.1 loading data
df = []

#load the header of analytics
analy_header = pd.read_csv("headers_analytics.csv")
#clean dependent matrix
etf_return = pd.read_csv("return/daily_return.csv")

#reading all files in directory and make a data clean
for file in glob.glob("analytics/*.csv"):
    f = pd.read_csv(file, header = None,names = analy_header)
    df.append(f)
    #data_description(f)

#1.2 data combination
#find correct time daily return and combine two data frame
#use directory to reserve data
reg_df = {}
trading_time = etf_return["Date"].tolist()    
for single_df in df:
    if not single_df.empty:
        time = str(single_df.iloc[0][0])
        time = datetime.datetime.strptime(time,'%Y%m%d').strftime("%Y-%m-%d")
        if time in trading_time:
            row = etf_return["Date"].str.contains(time)
            daily_return = etf_return[row].transpose()[1:]
            daily_return = daily_return.reset_index()
            daily_return.columns = ["ticker","daily_return"]
            reg_df[time] = pd.merge(single_df,daily_return,on = "ticker")
        else:
            print("no matching time", time)

#1.3 cleaning data        
#clean concatated data frame
#drop column with all na and rows with na
#drop outlier case
for time in reg_df:   
    reg_df[time] = reg_df[time][reg_df[time]["daily_return"] != 0]
    small = reg_df[time]["daily_return"] > -2
    large = reg_df[time]["daily_return"] < 2    
    reg_df[time] = reg_df[time][small & large]
    reg_df[time] = reg_df[time].dropna(axis = 1, how = "all")
    reg_df[time] = reg_df[time].dropna(axis = 0)   
    reg_df[time]["daily_return"] = pd.to_numeric(reg_df[time]["daily_return"])  

if os.path.isdir("data"):
    pass
else:
    os.mkdir("data")
    
for time in reg_df:
    reg_df[time].to_csv("data/%s.csv" %time, index =False)
 
#clean and update monthly return    
if os.path.isdir("data_monthly"):
    pass
else:
    os.mkdir("data_monthly")
     
close = pd.read_csv("return/close.csv")
new_close = close.drop("Date",axis = 1)
start_date = datetime.datetime.strptime(close["Date"][1],"%Y-%m-%d")
this_month = start_date.month
start_point = 1
end_point = 0
for i in range(len(close)):
    date = datetime.datetime.strptime(close["Date"][i],"%Y-%m-%d")
    if date.month != this_month:
        end_point = i - 1
        monthly_return = (new_close.iloc[start_point]/new_close.iloc[end_point] - 1)*100
        monthly_return = monthly_return.transpose()[1:]
        monthly_return = monthly_return.reset_index()
        monthly_return.columns = ["ticker","monthly_return"]
        file = "analytics/analytics_" + start_date.strftime("%Y%m%d") + ".csv"
        if os.path.isfile(file):
            single_df = pd.read_csv(file,header = None,names = analy_header)
            reg_df_month = pd.merge(single_df,monthly_return,on = "ticker")
            reg_df_month = reg_df_month[reg_df_month["monthly_return"] != 0]
            small = reg_df_month["monthly_return"] > -2
            large = reg_df_month["monthly_return"] < 2    
            reg_df_month= reg_df_month[small & large]
            reg_df_month = reg_df_month.dropna(axis = 1, how = "all")
            reg_df_month = reg_df_month.dropna(axis = 0)   
            reg_df_month["monthly_return"] = pd.to_numeric(reg_df_month["monthly_return"])            
            reg_df_month.to_csv("data_monthly/%s.csv" %start_date.strftime("%Y-%m"), index =False)
        start_point = i
        start_date = date
        this_month = date.month