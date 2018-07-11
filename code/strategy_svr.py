from strategy import selecting
import numpy as np
import pandas as pd
from env_context import context
import warnings
warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system.
warnings.filterwarnings('ignore')


# It's a pipeline to filter stocks quickly based on some features.
@selecting
def select_stocks(symbols, context_dict, f_calendar, top_per, bottom_per, instnum=5):
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

    def wrapper(test_y, long_position, short_position, *args):
        weight_new = pd.Series(np.zeros(test_y.shape[0]), index=test_y.index)
        flag = True
        pool_long, pool_short = order_case(test_y, *args)
        if (len(pool_long) > 0) & (long_position != 0):
            weight_new.loc[pool_long] = long_position / len(pool_long)
            # print(len(pool_long))
        if (len(pool_short) > 0) & (short_position != 0):
            weight_new.loc[pool_short] = short_position / len(pool_short)
            # print(len(pool_short))
        if (len(pool_long) <= 0) & (len(pool_short) <= 0):
            # equally-weighted portfolio with all stocks in test_y
            weight_new.loc[test_y.index.values] = 1.0 / np.shape(test_y)[0]

        if weight_new.empty:
            flag = False
        return weight_new, flag

    return wrapper


#  It's used to order stocks
@fix_stock_order
def order_method(test_y, context_dict, cur_date, remove, bottom_thre=1.0, top_thre=0.0, ):

    pool_short = []
    pool_long = []
    # case 1: only long position
    pool1 = test_y[test_y > test_y.quantile(q=1 - bottom_thre)]
    pool2 = test_y[test_y <= test_y.quantile(q=1 - top_thre)]
    if len(list(set(pool1.index.values).intersection(set(pool2.index.values)))) > 0:
        temp = pool2.loc[pool1.index]
        pool_long = temp[temp > 0].index.values
        # pool_long =temp.index.values
    # add short position
    pool1 = test_y[test_y <= test_y.quantile(q=bottom_thre)]
    pool2 = test_y[test_y > test_y.quantile(q=top_thre)]
    if len(list(set(pool1.index.values).intersection(set(pool2.index.values)))) > 0:
        temp = pool2.loc[pool1.index]
        pool_short = temp[temp < 0].index.values
        # pool_short = temp.index.values

    # case ??
    # indicator = test_y.loc[:,~np.isin(test_y.columns,remove)]
    # flag = indicator.apply(layer,args=(1-bottom_thre,1-top_thre,),axis =0)
    #
    # pool = flag[flag.sum(axis=1) == np.shape(indicator)[1]].index.values

    return pool_long, pool_short


if __name__ == "__main__":
    # parameters initialization
    #variable_list = ['quant_technical_st', 'quant_technical_it', 'quant_sentiment_iv', 'quant_fundamental_pe',
    # 'quant_technical_lt','quant_global_sector', 'quant_global_country']
    variable_list = ['quant_technical_st', 'quant_technical_it', 'quant_technical_lt', 'quant_sentiment_pc',
                     'quant_sentiment_si', 'quant_sentiment_iv', 'quant_fundamental_pb', 'quant_fundamental_pe',
                     'quant_fundamental_pcf', 'quant_fundamental_div', 'quant_global_sector', 'quant_global_country']

    leverage = 0.95
    long_position = 1.5
    short_position = 1 - long_position
    # end_day = -1 # -1 means till today
    start_day = '2012-01-01'
    trading_days = 252.0
    interest_rate = 0.0
    horizon = 21 * 1
    freq = 21 * 1  # rebalance monthly
    roll = 12  # rolling in 6 months
    ben = 'ACWI'  # benchmark
    model_name = 'SVR'
    relative = False
    bottom_thre = 0.1
    top_thre = 0.0

    # Back-test initialization
    context = context(start_day, leverage, long_position, short_position, \
                      interest_rate, trading_days, variable_list, freq)
    # Form training set and you just need run once and after that you can comment it until you change
    #  variable list or other parameters.
    # context.generate_train(horizon, relative, ben, normalize=True)
    # Name the results using parameters and so on
    address = 'etf_' + 'long' + str(long_position) + '_short' + str(short_position)[:5] + '_' + model_name + '_' + str(
        bottom_thre) + '_' + str(roll) + '.csv'
    context.back_test(ben, horizon, model_name, address, select_stocks, order_method,\
                      bottom_thre=bottom_thre, top_thre=top_thre, roll=roll)