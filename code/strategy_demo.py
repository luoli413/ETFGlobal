from strategy import selecting,fix_stock_order
import numpy as np
import pandas as pd
from env_context import context
import ftp_download as ftp

# It's a pipeline to filter stocks quickly based on some features.
@selecting
def select_stocks(symbols,context_dict, f_calendar, top_per, bottom_per,instnum = 5):
    keys = ['west_instnum_FY1','west_herdCV_FY1'] #'herdIndex']

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

#  It's used to order stocks
# TODO：allow shorting
@fix_stock_order
def order_method(test_y,context_dict,cur_date,remove,bottom_thre=1.0,top_thre=0.0,):
    # case 1
    pool1 = test_y[test_y >= test_y.quantile(q=1 - bottom_thre)]
    pool2 = test_y[test_y <= test_y.quantile(q=1 - top_thre)]
    pool = pool2.loc[pool1.index].index.values

    # case 2
    # pool = test_y[test_y <= -0.05]
    # pool = test_y[np.abs(test_y) < 0.05]
    # pool = test_y[test_y >= 0.05]

    # indicator = test_y.loc[:,~np.isin(test_y.columns,remove)]
    # case 3
    # flag = indicator.apply(ind_layer,args=(context_dict['industry_d2'].loc[cur_date,:],\
    #                                              1-bottom_thre,1-top_thre,),axis = 0)
    # case 4
    # flag = indicator.apply(layer,args=(1-bottom_thre,1-top_thre,),axis =0)
    #
    # pool = flag[flag.sum(axis=1) == np.shape(indicator)[1]].index.values

    return pool

if __name__ == "__main__":
    # ftp.download_data('2017-12-30')
    ftp.close_processing()
    # parameters initialization
    variable_list = ['quant_technical_st','quant_technical_it', 'quant_technical_lt']
    leverage = 1.0
    # end_day = -1 # -1 means till today
    start_day = '2013-01-01'
    trading_days = 252.0
    horizon = 21*2
    freq = 21 #rebalance monthly
    roll = 6
    ben = 'ben'
    model_name = 'Random'
    relative = True
    top_per = 20
    bottom_per = top_per + 20

    # # back-test initialization
    context = context(start_day,leverage,trading_days,variable_list,)
    # form training set and you just need run once and after that you can comment it until you change variable list.
    # context.generate_train(variable_list,horizon, relative, ben, normalize=True)
    # name the results using parameters and so on
    address = 'test_etf' + '_' + str(top_per) + '_' + str(bottom_per) + '_' + model_name + '.csv'
    context.back_test(ben, horizon, freq, model_name, address, select_stocks, order_method, \
                      top_per=top_per, bottom_per=bottom_per, roll=roll)

