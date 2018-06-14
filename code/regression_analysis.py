# -*- coding: utf-8 -*-

import datetime
import os.path
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV

def get_data_by_period(start_date,end_date,daily_or_monthly = "daily"):
    data =pd.DataFrame() 
    if daily_or_monthly == "daily":
        start = datetime.datetime.strptime(start_date,"%Y-%m-%d").date()
        end = datetime.datetime.strptime(end_date,"%Y-%m-%d").date()
        for t in range((end-start).days + 1):
            date = start + datetime.timedelta(t)
            file = "data/" + date.strftime("%Y-%m-%d") + ".csv"
            if os.path.isfile(file):
                data = data.append(pd.read_csv(file))
    elif daily_or_monthly == "monthly":
        start = datetime.datetime.strptime(start_date,"%Y-%m-%d")
        end = datetime.datetime.strptime(end_date,"%Y-%m-%d")
        months_diff = (end.year - start.year)* 12 + end.month - start.month 
        for t in range(months_diff+1):
             month = (start.month + t) % 12
             year = start.year + int((start.month + t) / 12)
             if month == 0:
                 month = 12
                 year = year - 1
             date = datetime.date(year,month,1)
             file = "data_monthly/" + date.strftime("%Y-%m") + ".csv"
             if os.path.isfile(file):
                 data = data.append(pd.read_csv(file))
        
    else:
        print("something wrong")
    return data         


features =['quant_technical_st','quant_technical_it', 'quant_technical_lt', 'quant_sentiment_pc', 'quant_sentiment_si', 'quant_sentiment_iv',
       'quant_fundamental_pe', 'quant_fundamental_pcf', 'quant_fundamental_pb',
       'quant_fundamental_div', 'quant_global_sector', 'quant_global_country', 'quant_quality_liquidity', 'quant_quality_diversification',
       'quant_quality_firm']

#A quick view on features
def hist_on_features(data):
    fig, axs = plt.subplots(3, 5)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for i in range(3):
        for j in range(5):
            axs[i,j].hist(data[features[i* 5 + j]])
            
def test_collinearity(data):
    cov=data.cov()
    eig_val, eig_vec = np.linalg.eig(cov)
    print(eig_val)
    for num in eig_val:
        if num < 1 and num > -1:
            print("Small eigenvalue, indicate multicollinearity")
            return True
    return False  

def ordinary_linear(y,x): 
    model = sm.OLS(y,x)
    print(model.fit().summary())
    feature = []
    for key,value in model.fit().pvalues.items():
        if value < 0.05 and key != "const":
            feature.append(key)
    return feature


#stepwise regression
#df: dataframe of data
#Y: data of the independent data
#features: the features column of the dataframe
#alpha: setting alpha to limiited the minimum pvalue  
    
def stepwise(df,Y,features,alpha):
    feature_col = list(features)
    length  =  len(feature_col)
    final_feature = []
    for i in range(length): 
        pvalue_min = 1
        column_min = ""   
        for feature in feature_col:
            temp_feature  = final_feature + [feature]
            x = sm.add_constant(df[temp_feature])
            model = sm.OLS(Y,x)
            pvalue = model.fit().pvalues[i+1] 
            if pvalue < pvalue_min and pvalue < alpha:
                pvalue_min = pvalue
                column_min = feature
            
        if column_min != "":     
            feature_col.remove(column_min)
            final_feature.append(column_min)
        else:
            break
    X = sm.add_constant(df[final_feature])
    model =  sm.OLS(Y,X)
    return final_feature
    
def Lasso_feature_select(y,x):
    lassocv = LassoCV(fit_intercept=True)
    model = lassocv.fit(x,y)
    print("LASSO feature Selection:")
    print("Best fit alpha under Lasso Cross Validation: %f" %model.alpha_)
    print(np.array(features)[model.coef_[1:] != 0])
    return np.array(features)[model.coef_[1:] != 0]
            
start_date = "2012-03-01"
end_date = "2012-04-01"
daily_data = get_data_by_period(start_date,end_date)

start_month = "2012-03-01"
end_month = "2012-06-01"
monthly_data = get_data_by_period(start_month,end_month,"monthly")

hist_on_features(daily_data) 
hist_on_features(monthly_data)    
    
x = sm.add_constant(daily_data[features])
y = daily_data["daily_return"]

print("ols result:")
print(ordinary_linear(y,x))
print("Stepwise regression model result:")
print(stepwise(daily_data,y,features,0.05))    
Lasso_feature_select(y,x)    

monthly_y = monthly_data["monthly_return"]
monthly_x = sm.add_constant(monthly_data[features])

print("ols result:")
print(ordinary_linear(monthly_y,monthly_x))
print("Stepwise regression model result:")
print(stepwise(monthly_data,monthly_y,features, 0.05))
Lasso_feature_select(monthly_y,monthly_x)



