from strategy import selecting
import numpy as np
import pandas as pd
from env_context import context
import warnings
warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system.
warnings.filterwarnings('ignore')


# It's a pipeline to filter stocks quickly based on some features.
@selecting
def select_stocks(symbols,context_dict, f_calendar, *args,**kwargs):
    # keys = ['west_instnum_FY1','west_herdCV_FY1'] #'herdIndex']

    # indexing = f_calendar[-1]
    # for i in keys:
    #     fd = context_dict[i]
    #     fd = fd.loc[:, symbols]
    #     if i == 'west_instnum_FY1':
    #         temp = fd.columns[fd.loc[indexing, :] >= instnum].values
    #     else:
    #         temp_fd = fd.loc[indexing, :]
    #         temp1 = temp_fd[temp_fd > np.nanpercentile(temp_fd, (100 - bottom_per))].index.values
    #         temp2 = temp_fd[temp_fd < np.nanpercentile(temp_fd, 100 - top_per)].index.values
    #         temp = list(set(temp1).intersection(set(temp2)))
    #     symbols = list(set(symbols).intersection(set(temp)))
    return symbols

def fix_stock_order(order_case):

    def wrapper(test_y,long_position,short_position,*args,**kwargs):
        weight_new = pd.Series(np.zeros(test_y.shape[0]), index=test_y.index)
        flag = True
        pool_long,pool_short = order_case(test_y,*args,**kwargs)
        if len(pool_long)>0:
            weight_new.loc[pool_long] = long_position / len(pool_long)
            # print(len(pool_long))
        if len(pool_short)>0:
            weight_new.loc[pool_short] = short_position / len(pool_short)
            # print(len(pool_short))
        if len(pool_long)<=0 & len(pool_short) <= 0:
            # equally-weighted portfolio with all stocks in test_y
           weight_new.loc[test_y.index.values] = 1.0 / np.shape(test_y)[0]

        if weight_new.empty:
            flag = False
        return weight_new,flag

    return wrapper

#  It's used to order stocks
@fix_stock_order
def order_method(test_y,context_dict,cur_date,thre):

    pool_short = []
    pool_long = []
    # case 1: only long position
    pool1 = test_y[test_y > test_y.quantile(q=1 - thre['long_thre'][1])]
    pool2 = test_y[test_y <= test_y.quantile(q=1 - thre['long_thre'][0])]
    if len(list(set(pool1.index.values).intersection(set(pool2.index.values))))>0:
        temp = pool2.loc[pool1.index]
        pool_long = temp[temp>0].index.values
    # add short position
    pool1 = test_y[test_y <= test_y.quantile(q= thre['short_thre'][1])]
    pool2 = test_y[test_y > test_y.quantile(q= thre['short_thre'][0])]
    if len(list(set(pool1.index.values).intersection(set(pool2.index.values)))) > 0:
        temp = pool2.loc[pool1.index]
        pool_short = temp[temp<0].index.values

    # case ??
    # indicator = test_y.loc[:,~np.isin(test_y.columns,remove)]
    # flag = indicator.apply(layer,args=(1-bottom_thre,1-top_thre,),axis =0)
    #
    # pool = flag[flag.sum(axis=1) == np.shape(indicator)[1]].index.values

    return pool_long,pool_short


if __name__ == "__main__":

    # parameters initialization
    variable_list = ['quant_technical_st','quant_technical_it','quant_technical_lt','quant_sentiment_iv',\
                     'quant_sentiment_si','quant_sentiment_pc','quant_fundamental_pe','quant_fundamental_pb',\
                     'quant_fundamental_pcf','quant_fundamental_div','quant_global_sector','quant_global_country',\
                     'quant_quality_diversification','quant_quality_firm','quant_quality_liquidity']

    start_day = '2012-01-01'
    horizon = 21*1
    freq = 21*1  # rebalance monthly
    roll = 36 # rolling in x months
    ben = 'ACWI' # benchmark
    model_name = 'StepWise'
    relative = True

    # Back-test initialization
    context = context(ben, start_day,variable_list,freq,)
    # Form training set and you just need run once and after that you can comment it until you change
    #  variable list or other parameters.
    # context.generate_train(horizon, relative, normalize=True,method='98%shrink')
    context.feature_selection(horizon,method=model_name,freq=freq,model_name = model_name,\
                              select_method=select_stocks,roll=roll)