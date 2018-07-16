# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import strategy as strats
import datetime
path = os.getcwd()
data_path = os.path.join(path + '\\data\\')
ana_path = os.path.join(path+'\\analytics\\')
res_path = os.path.join(path+ '\\res\\')

def compute_indicators(df,ben,save_address,trading_days, required=0.00, whole=1):
    # columns needed
    if ben == None:
        ben = str(ben)
    col = [ben, 'nav', 'rebalancing','summary_score', 'stoploss', 'Interest_rate']

    df_valid = df.loc[:, col]
    start_balance = df.index[df['rebalancing'] == 1][0]
    df_valid = df_valid[pd.to_datetime(df_valid.index) >= \
                        pd.to_datetime(start_balance)]
    # average socre of models in training set
    df_valid['ave_score'] = df['summary_score'].expanding(min_periods=1).apply(lambda x: np.nanmean(x))
    df_valid['summary_score'] = df_valid['summary_score'].fillna(method='ffill')
    # daily return
    df_valid['return'] = (df['nav'] - df['nav'].shift(1))/ df['nav'].shift(1)
    # benchmark_net_value
    df_valid[ben] = df_valid[ben] / df_valid[ben].iloc[0]

    # benchmark_return
    df_valid[ben+'_return'] = (df_valid[ben] -
                                    df_valid[ben].shift(1)) / \
                                   df_valid[ben].shift(1)
    # Annualized return
    df_valid['Annu_return'] = df_valid['return'].expanding(min_periods=1).mean() * trading_days
    # Volatility
    df_valid.loc[:, 'algo_volatility'] = df_valid['return'].\
                                             expanding(min_periods=1).std() * np.sqrt(trading_days)
    temp = df_valid['return'] - \
                              df_valid['Interest_rate'] / trading_days / 100
    temp_rela = df_valid['return'] - df_valid[ben+'_return']

    def ratio(x):
        return np.nanmean(x) / np.nanstd(x)

    # sharpe ratio
    df_valid.loc[:, 'sharpe'] = temp.expanding(min_periods=1).apply(ratio) \
                                * np.sqrt(trading_days)
    # sharpe of benchmark
    temp = df_valid[ben+'_return'] - \
                              df_valid['Interest_rate'] / trading_days / 100
    df_valid.loc[:,ben+'_sharpe'] = temp.expanding(min_periods=1).apply(ratio) \
                                * np.sqrt(trading_days)
    # information ratio
    df_valid.loc[:, 'IR'] = temp_rela.expanding().apply(ratio) \
                            * np.sqrt(trading_days)

    # Sortino ratio
    def modify_ratio(x, re):
        re /= trading_days
        ret = np.nanmean(x) - re
        st_d = np.nansum(np.square(x[x < re] - re)) / x[x < re].size
        return ret / np.sqrt(st_d)

    df_valid.loc[:, 'sortino'] = df_valid['return'].expanding().\
                                     apply(modify_ratio, args=(required,)) * np.sqrt(trading_days)
    # loss probability
    wins = np.where(df_valid['return'] < 0, 1.0, 0.0)
    df_valid.loc[:, 'loss_rate'] = wins.cumsum() / \
                                  pd.Series(wins, index=df_valid.index).expanding(min_periods=2).apply(len)
    # Transfer infs to NA
    df_valid.loc[np.isinf(df_valid.loc[:, 'sharpe']), 'sharpe'] = np.nan
    df_valid.loc[np.isinf(df_valid.loc[:, ben+'_sharpe']), ben+'_sharpe'] = np.nan
    df_valid.loc[np.isinf(df_valid.loc[:, 'IR']), 'IR'] = np.nan
    # hit_rate
    wins = np.where(df_valid['return'] >= df_valid[
        ben+'_return'], 1.0, 0.0)
    df_valid.loc[:, 'hit_rate'] = wins.cumsum() / \
                                  pd.Series(wins,index =df_valid.index).expanding(min_periods=2).apply(len)
    # 95% VaR
    df_valid['VaR'] = -df_valid['return'].expanding().quantile(0.05) * \
                      np.sqrt(trading_days)
    # 95% CVaR
    df_valid['CVaR'] = -df_valid['return'].expanding().apply(lambda x: \
                np.nanmean(x[x < np.nanpercentile(x,5)])) * np.sqrt(trading_days)


    if whole == 1:
        # max_drawdown
        def exp_diff(x, type):
            if type == 'dollar':
                xret = x.expanding().apply(lambda xx: (xx[-1] - xx.max()))
            else:
                xret = x.expanding().apply(lambda xx: (xx[-1] - xx.max()) / xx.max())
            return xret
            # dollar
            #     xret = exp_diff(df_valid['cum_profit'],'dollar')
            #     df_valid['max_drawdown_profit'] = abs(pd.expanding_min(xret))
            # percentage

        xret = exp_diff(df_valid['nav'], 'percentage')
        df_valid['max_drawdown_ret'] = abs(xret.expanding().min())

        # max_drawdown_duration:
        # drawdown_enddate is the first time for restoring the max
        def drawdown_end(x, type):
            xret = exp_diff(x, type)
            minloc = xret[xret == xret.min()].index[0]
            x_sub = xret[xret.index > minloc]
            # if never recovering,then return nan
            try:
                return x_sub[x_sub == 0].index[0]
            except:
                return np.nan

        def drawdown_start(x, type):
            xret = exp_diff(x, type)
            minloc = xret[xret == xret.min()].index[0]
            x_sub = xret[xret.index < minloc]
            try:
                return x_sub[x_sub == 0].index[-1]
            except:
                return np.nan

        df_valid['max_drawdown_start'] = pd.Series()
        df_valid['max_drawdown_end'] = pd.Series()
        df_valid['max_drawdown_start'].iloc[-1] = drawdown_start(
            df_valid['nav'], 'percentage')
        df_valid['max_drawdown_end'].iloc[-1] = drawdown_end(
            df_valid['nav'], 'percentage')
    if os.path.isdir("res"):
        pass
    else:
        os.mkdir("res")
    df_valid.to_csv(save_address)

class context(object):

    def get_data_monthly(self,method):

        df = pd.DataFrame()
        close_df = self.context_dict['close']
        close_df = close_df[close_df.index >= pd.to_datetime(self.start_day)]
        if self.end_day != -1:
            close_df = close_df[close_df.index <= pd.to_datetime(self.end_day)]

        if method=='last_date_monthly':
            f_calendar = pd.Series(close_df.index.values)

            def date_transfer(x,unit='d'):
                if unit=='d':
                    x = datetime.datetime.strptime(str(x)[:10], "%Y-%m-%d")
                if unit=='m':
                    x = datetime.datetime.strptime(str(x)[:10], "%Y%m")
                return x

            date_df = pd.DataFrame(f_calendar.apply(date_transfer),columns=['Date'])
            date_df['monthly_date'] = date_df['Date'].apply(lambda x: str(x)[:8])
            f_calendar = date_df.groupby('monthly_date')['Date'].max()
            for date in f_calendar.values:
                file = ana_path + 'analytics_' + \
                       datetime.datetime.strptime(str(date)[:10],'%Y-%m-%d').strftime("%Y%m%d") + '.csv'
                if os.path.isfile(file):
                    ana_df = pd.read_csv(file, header=None, )
                    if df.empty:
                        df = ana_df
                    else:
                        df = pd.concat([df, ana_df])


        df.columns = ['Date', 'tic'] + list(self.feature_name)
        df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
        df.sort_values(['Date'], inplace=True)
        df.set_index(['Date'], inplace=True, drop=True)
        return df

    def __init__(self,ben,start_day,leverage,long_position,short_position,interest_rate,\
                 trading_days,f_list,freq,daily =False,end_day=-1,method='monthly'):

        headers = pd.read_csv(ana_path+'headers_analytics.csv')
        self.feature_name = headers.columns.values[2:]
        self.ben = ben
        self.start_day = start_day
        self.end_day = end_day
        self.trading_data_list = ['close','volume']
        self.leverage = leverage
        self.long_position = long_position
        self.short_position = short_position
        self.trading_days = trading_days
        self.variable_list = f_list
        self.interest_rate = interest_rate
        self.freq = freq
        self.daily = daily

        # import close data
        self.context_dict = dict()
        for i in self.trading_data_list:
            temp = pd.read_csv(os.path.join(data_path + i + '.csv'))
            temp_col = temp.columns.values
            temp_col[0] = 'Date'
            temp.columns = temp_col
            try:
                temp['Date'] = pd.to_datetime(temp['Date'],format ='%m/%d/%Y')
            except:
                try:
                    temp['Date'] = pd.to_datetime(temp['Date'], format='%Y-%m-%d')
                except:
                    temp['Date'] = pd.to_datetime(temp['Date'], format='%Y%m%d')
            temp.sort_values(['Date'], inplace=True)
            temp.set_index(['Date'], drop=True, inplace=True)
            self.context_dict[i] = temp
        print('import trading data completed!')
        # import features
        df = self.get_data_monthly(method)
        #  column which contains char to num
        degree = df['quant_grade'].value_counts().index.values
        degree.sort()

        s = 1.0
        for grade in degree:
            df.loc[df['quant_grade']==grade,'quant_grade'] = s
            s+=1

        self.features = df.loc[:,['tic']+self.variable_list]
        self.features.dropna(inplace=True)
        self.features.to_csv('feature.csv')
        print('import features completed!')


    def book(self,df,weights,weight_new_temp=None,stats_summary = None,rebalance_flag = False):

        def record_weights(df, i, weights):
            return_vector = (df.iloc[i, :][weights.columns] - df.iloc[i - 1, :][weights.columns]) \
                            / df.iloc[i - 1, :][weights.columns]
            #   if the stock had not listed in market or no data available,we need set the return as zero
            return_vector[return_vector.isnull()] = 0.0
            sum_return = np.dot(return_vector, weights.iloc[i - 1, :].values)
            every_re = np.multiply(weights.iloc[i - 1, :], (return_vector + 1))
            weights.iloc[i, :] = every_re / (1 + sum_return)

            return weights

        def record_return(df, i, reb_index, weight_new, leverage, trading_days,interest_rate,daily=False,):
            if daily:
                reb_index = i - 1

            return_vector = (df.iloc[i, :][weight_new.index.values] - \
                             df.iloc[reb_index, :][weight_new.index.values]) / \
                            df.iloc[reb_index, :][weight_new.index.values]
            return_vector[return_vector.isnull()] = 0.0  # no trade,then no return
            cum_return = np.dot(return_vector, weight_new.values)
            df['nav'].iloc[i] = df['nav'].iloc[reb_index] * (\
                1 + cum_return * leverage + (1 - leverage) * (i - reb_index)* interest_rate / trading_days / 100)
            return df

        if rebalance_flag:
            self.weight_new = weight_new_temp
            print(str(df.index[self.s - 1])[:10], \
                  len(self.weight_new[self.weight_new != 0]), 'stocks in portfolio')

            weights.loc[df.index[self.s - 1], :] = 0.0
            weights.loc[df.index[self.s - 1], self.weight_new.index] = self.weight_new.values
            df['rebalancing'].iloc[self.s - 1] = 1
            if stats_summary is None:
                pass
            elif isinstance(stats_summary, pd.Series):
                df['summary_score'].iloc[self.s - 1] = stats_summary['score']
            else:
                df['summary_score'].iloc[self.s - 1] = stats_summary
            self.reb_index = self.s - 1
            rebalance_flag = False

        if len(self.weight_new)>0:
            df = record_return(df, self.s, self.reb_index, self.weight_new, \
                                   self.leverage, self.trading_days,self.interest_rate,self.daily)
            weights = record_weights(df, self.s, weights)

        return df,weights,rebalance_flag

    def generate_train(self,horizon,relative,normalize = False,*args,**kwargs):

        v_list = ['tic'] + self.variable_list
        fd_data = self.features[v_list].copy()
        fd_data.replace([np.inf,-np.inf], np.nan,inplace=True) # get rid of infs
        p_data = self.context_dict['close'].copy()
        if self.end_day!=-1:
            end_day = pd.to_datetime(self.end_day)
            p_data = p_data[pd.to_datetime(p_data.index) <= end_day]
            fd_data = fd_data[pd.to_datetime(fd_data.index) <= end_day]
        f_calendar = fd_data.index.drop_duplicates()
        cols = p_data.columns
        symbols = cols[cols!= self.ben].values
        fd_data = fd_data[np.isin(fd_data['tic'],symbols)]

        # ===== Deal with Y: future returns
        returns = (p_data.shift(-horizon) - p_data) / p_data
        if relative & (self.ben is not None):
            ben = returns[self.ben]
            returns = pd.DataFrame(np.subtract(np.array(returns),
                                               np.array(ben).reshape(len(ben), 1)),
                                   index=returns.index, columns=returns.columns)

        def normalized(x,method = '3sigma'):
            x = pd.Series(x)
            clean_x = x[~x.isnull()]
            if len(clean_x) > 3:
                if method=='3sigma':
                    miu = np.nanmedian(clean_x)
                    sigma = np.nanstd(clean_x)
                    if sigma > 0:
                        x = (x - miu) / sigma
                        x[(~x.isnull()) & (x > 3)] = 3
                        x[(~x.isnull()) & (x < -3)] = -3
                if method =='98%shrink':
                    head = clean_x.quantile(0.99)
                    tail = clean_x.quantile(0.01)
                    x[x > head] = head
                    x[x < tail] = tail
                    sigma = np.nanstd(clean_x)
                    miu = np.nanmean(clean_x)
                    x = (x - miu) / sigma
            # print(len(x))
            return x

        # Deal with Xs: normalize in all stocks each quarter
        if normalize:
            for time in f_calendar.values:
                if time in fd_data.index:
                    temp = fd_data.loc[time, self.variable_list]
                    # print(temp.shape)
                    if not isinstance(temp,pd.Series):
                        fd_data.loc[time, self.variable_list] \
                            = np.apply_along_axis(normalized, 0, np.array(temp),kwargs)

        def append_y(x, re_st,):
            dateindex = pd.to_datetime(re_st.index, infer_datetime_format=True)
            temp = re_st.index[dateindex >= pd.to_datetime(x)]

            if len(temp) > 0:
                return re_st.loc[temp[0]]
            else:
                return np.nan

        # def apply_append_y(x,returns,train):
        #     re_st = returns[x]
        #     rdq = pd.Series(train.loc[train['tic'] == x, :].index, \
        #                     index=train.loc[train['tic'] == x, :].index)
        #     if len(rdq) > 0:
        #         train.loc[train['tic'] == x, 'y'] = rdq.apply(append_y, args=(re_st,))
        #         print(x)
        #     return train

        train = fd_data.copy()
        train.loc[:, 'y'] = pd.Series()
        # train = pd.Series(symbols).apply(apply_append_y, args=(returns, train,))
        for tics in symbols:
            re_st = returns[tics]
            rdq = pd.Series(train.loc[train['tic'] == tics, :].index, \
                            index=train.loc[train['tic'] == tics, :].index)
            if len(rdq) > 0:
                train.loc[train['tic'] == tics, 'y'] = rdq.apply(append_y, args=(re_st,))
                print(tics)
        self.train = train
        train.to_csv('trains.csv')
        print(train.shape)
        print('generating train completed!')

    def extract_train(self,cur_date,horizon,select_method,roll=-1,*args,**kwargs):

        v_list = ['tic', 'y'] + self.variable_list
        # train = self.train[v_list].copy()

        # # read data from file
        train = pd.read_csv('trains.csv')
        train['Date'] = pd.to_datetime(train['Date'],format='%Y-%m-%d')
        train.sort_values(['Date'],inplace=True)
        train.set_index('Date',drop=True,inplace=True)
        train = train[v_list].copy()

        train = train[pd.to_datetime(train.index) < cur_date]
        bool = False
        if not train.empty:
            f_calendar = train.index.drop_duplicates()

            # rolling
            if (roll == -1) or (roll+1 > len(f_calendar)):
                pass
            else:
                f_calendar = f_calendar[-(roll+1):]

            if len(f_calendar)>=2:

                symbols = select_method(self.context_dict, f_calendar, *args,**kwargs)
                train_all = train[np.isin(train['tic'], symbols)]
                # print(train_all.index.drop_duplicates())
                train_calendar = f_calendar[(pd.to_datetime(f_calendar) -\
                                             cur_date).days <= -(horizon/21) * 31.0] #lag back

                if len(list(set(train_calendar).intersection(set(train_all.index.drop_duplicates()))))>0:

                    train = train_all[np.isin(train_all.index.values,train_calendar.values)].copy()
                    train.dropna(how='any', axis=0,inplace=True)
                    self.y_train = train['y']
                    self.x_train = train[self.variable_list]

                    # x_test = pd.DataFrame()
                    # s=0
                    # for tic in symbols:
                    #     # Some stocks may not update lately so we still use the latest but old data available
                    #     test_temp = train_all[train_all['tic'] == tic]
                    #     if len(test_temp)>0:# fd_data does not cover some stocks in certain early date
                    #         test_date = test_temp.index[-1]
                    #         # print(test_date)
                    #         if s == 0:
                    #             x_test = test_temp.loc[test_date,:]
                    #         else:
                    #             x_test = pd.concat([x_test,
                    #                                 test_temp.loc[test_date,:]],axis=1)
                    #         s += 1
                    # x_test = x_test.T
                    ## generate x_test
                    if f_calendar[-1] in train_all.index:  # some stocks may not be traded as this moment.
                        temp = train_all.loc[f_calendar[-1], :]
                        if isinstance(temp, pd.Series):
                            temp = temp.to_frame().T
                        x_test = temp[np.isin(temp['tic'], symbols)]
                        # s = np.shape(x_test)[0]
                        # print(s,'stocks completed in x_test')
                        x_test.set_index(['tic'], drop=True, inplace=True)  # need know data belongs to whom
                        x_test.drop(['y'], axis=1, inplace=True)
                        x_test.dropna(how='any', axis=0, inplace=True)
                        self.x_test = x_test

                        bool = True

        return bool

    def back_test(self,horizon,model_name,address,select_method,order_method,roll=-1,*args,**kwargs):
        # initial setting
        df = self.context_dict['close'].copy()
        symbols = self.context_dict['close'].columns[:]
        symbols = symbols[symbols!= self.ben]
        stock_num = len(symbols)
        back_testing = df.index[pd.to_datetime(df.index) >= pd.to_datetime(self.start_day)]
        df = df.loc[back_testing.values, :]
        unit = np.full((len(df.index), 1), 1)[:, 0]

        df['rebalancing'] = pd.Series()
        df['stoploss'] = pd.Series()
        df['nav'] = pd.Series(unit, index=df.index)
        df['Interest_rate'] = pd.Series(np.full((len(df.index),), self.interest_rate),index=df.index)
        df['summary_score'] = pd.Series()
        self.weight_new = []
        weight_new_temp = None
        summary = None
        flag = False
        # max_new = []  # for computing max_drawdown
        unit = np.full((len(df.index), stock_num), 0)
        weights = pd.DataFrame(unit, index=df.index, columns=symbols)
        self.reb_index = 0
        self.s = 0  # counting date

        # ============================= Enter Back-testing ===================================
        for cur_date in back_testing.values:

            cur_date = pd.to_datetime(cur_date)
            # rebalance in a fixed frequency in freq rate
            if self.s>0:# begin to rebalance at least after the second recordings
                if np.mod(self.s, self.freq) == 0:

                    if self.extract_train(cur_date,horizon,select_method,roll=roll,*args,**kwargs):
                        if np.shape(self.x_test)[0]>0:
                            test_y,summary = strats.model(model_name, self.x_train, self.y_train, self.x_test)
                            weight_new_temp,flag = order_method(test_y,self.long_position,self.short_position,\
                                                        self.context_dict,cur_date,*args,**kwargs)
            df,weights,flag = self.book(df,weights,weight_new_temp,summary,flag)
            self.s += 1

        if np.shape(df[df['rebalancing']==1])[0]>1:
            compute_indicators(df, self.ben,res_path+'perf_'+address,self.trading_days)
            weights.to_csv(res_path+'weights_' + address)
        else:
            print('no rebalance, no performnace.')

        print('back_test completed!')

    # def test_eps_surprise(self,address,select_method,top_per,bottom_per,thre=0.05):
    #     v_list = ['tic',] + self.variable_list
    #     train = pd.read_csv('trains.csv')
    #     train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
    #     train.sort_values(['Date'], inplace=True)
    #     train.set_index('Date', drop=True, inplace=True)
    #     train = train[v_list].copy()
    #     calendar = train.index.drop_duplicates().values
    #     surprise_CV = pd.DataFrame()
    #     s = 0
    #     monthly_ratio = pd.DataFrame(index = calendar,columns=['positive','negative'])
    #     for cur_date in calendar:
    #         print('.', end='', flush=True)
    #         cur_calendar = calendar[calendar <= cur_date]
    #         symbols = select_method(self.context_dict, cur_calendar,top_per,bottom_per)
    #         if len(symbols)>0:# in case that symbols are empty
    #             # stats monthly
    #             temp = train[np.isin(train['tic'],symbols)].loc[cur_date,:]
    #             temp.dropna(how='any', axis=0, inplace=True)
    #             surprise_ratio = np.shape(temp[temp['eps_surprise_FTM'] > thre])[0] \
    #                              / np.shape(temp)[0]
    #
    #             neg_surprise_ratio = np.shape(temp[temp['eps_surprise_FTM'] < -thre])[0] / \
    #                                  np.shape(temp)[0]
    #             monthly_ratio.loc[cur_date,'positive'] = surprise_ratio
    #             monthly_ratio.loc[cur_date,'negative'] = neg_surprise_ratio
    #
    #             if s == 0:
    #                 surprise_CV = temp
    #             else:
    #                 surprise_CV = pd.concat([surprise_CV,temp],axis = 0)
    #         s += 1
    #
    #     # surprise_CV.dropna(how ='any',axis =0, inplace=True)
    #     surprise_ratio = np.shape(surprise_CV[surprise_CV['eps_surprise_FTM']>thre])[0]/\
    #         np.shape(surprise_CV)[0]
    #     neg_surprise_ratio = np.shape(surprise_CV[surprise_CV['eps_surprise_FTM']<-thre])[0]/\
    #         np.shape(surprise_CV)[0]
    #     print('\n',top_per,bottom_per)
    #     print(surprise_ratio,neg_surprise_ratio)
    #     monthly_ratio.to_csv('prob_'+address)
    #
    # def test_features_sig(self,model_name,cur_date,horizon,select_method,roll =-1):
    #     if self.extract_train(cur_date,horizon,select_method,top_per=0,bottom_per=100,roll=roll):
    #         temp,summary = strats.model(model_name, self.x_train, self.y_train, self.x_test)
    #     return summary

    def integrate_summary(self,horizon,freq,model_name,select_method,roll=-1):

        v_len = len(self.variable_list)
        integrate_summary = pd.DataFrame(np.zeros((v_len+4,2)),\
                        index = self.variable_list + ['const','f_test','score','missing_rebalance'],\
                                         columns=['p_value_average', 'num'])
        missing = []
        # def back_test(self, ben, horizon, freq, model_name,  ):
        # initial setting
        df = self.context_dict['close'].copy()

        back_testing = df.index[pd.to_datetime(df.index) >= pd.to_datetime(self.start_day)]

        s = 0  # counting date
        # ============================= Enter Back-testing ===================================
        for cur_date in back_testing.values:

            cur_date = pd.to_datetime(cur_date)
            # rebalance in a fixed frequency in freq rate
            if s > 0:  # begin to rebalance at least after the second recordings
                if np.mod(s, freq) == 0:

                    if self.extract_train(cur_date, horizon, select_method,roll=roll,):
                        if np.shape(self.x_test)[0] > 0:
                            test_y, summary = strats.model(model_name, self.x_train, self.y_train, self.x_test)
                            # weight_new_temp, flag = order_method(test_y, self.long_position, self.short_position, \
                            #                                      self.context_dict, cur_date, remove, bottom_thre,
                            #                                      top_thre)
                            temp = summary
                            # temp.set_index([0], drop=True, inplace=True)
                            integrate_summary.loc[temp.index.values, 'num'] += 1
                            ss = integrate_summary.loc[temp.index.values, 'num']
                            integrate_summary.loc[temp.index.values, 'p_value_average'] \
                                = (ss - 1.0) / ss * integrate_summary.loc[\
                                temp.index, 'p_value_average'] + 1.0 / ss * temp.iloc[:]
                            if 'f_test' not in temp.index.values:
                                missing.append(cur_date)
            s+=1
        integrate_summary = integrate_summary.astype('object')
        integrate_summary.at['missing_rebalance', 'num'] = missing
        integrate_summary.to_csv(res_path + 'inds_summary_all_' + str(horizon) +'_'+str(roll)+'.csv')

    # def feature_indus_filter(self,inds,):
    #     inds_num = []
    #     for ind in inds:
    #         inds_num.append(self.ind_dict[ind])
    #     calendar = self.features.index.drop_duplicates()
    #     print(calendar[-1])
    #     for time in calendar.values:
    #        tics = self.context_dict['industry_d2'].\
    #            loc[time,~np.isin(self.context_dict['industry_d2'].loc[time,:],inds_num)].index.values
    #        self.features.loc[time,:].loc[np.isin(self.features.loc[time,'tic'],tics),:] = np.nan
    #     self.features.dropna(how ='any',axis =0,inplace=True)
    #     print('filter industry completed!')