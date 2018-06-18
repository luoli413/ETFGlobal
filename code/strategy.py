import pandas as pd
import numpy as np

# Please add models here!!!
def model(model_name, train_x, train_y, test_x,alpha = 0.1):
    summary=[]
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    import sklearn
    import statsmodels.regression.linear_model as sm
    if model_name == 'Random':
        test_y = pd.Series(np.random.random_sample((len(test_x),)), index=test_x.index)
    if model_name == 'None':
        test_y = test_x.iloc[:,:]
    if model_name == 'MLPRegressor':
        mlp = MLPRegressor(hidden_layer_sizes=(20, 20))
        mlp.fit(train_x, train_y)
        y_pred = mlp.predict(test_x)
        test_y = pd.Series(y_pred, index=test_x.index)
    if model_name == 'Lasso':
        model = sklearn.linear_model.Lasso(0.001)
        lasso = model.fit(train_x, train_y)
        test_y = pd.Series(lasso.predict(test_x), index=test_x.index)

    if model_name == 'Ridge':
        model = sklearn.linear_model.Ridge(50)
        ridge = model.fit(train_x, train_y)
        test_y = pd.Series(ridge.predict(test_x), index=test_x.index)
    if model_name == 'SVR':
        svr_rbf = SVR(kernel='rbf', C=1, gamma=0.0001, epsilon=0.1)
        svr_rbf.fit(train_x, train_y)
        y_pred_rbf = svr_rbf.predict(test_x)
        test_y = pd.Series(y_pred_rbf, index=test_x.index)
    if model_name == 'StepWise':

        feature_col = list(train_x.columns.values)
        length = len(feature_col)
        final_feature = []
        for i in range(length):
            pvalue_min = 1
            column_min = ""
            for feature in feature_col:
                temp_feature = final_feature + [feature]
                x = sm.add_constant(train_x.loc[:,temp_feature])
                model = sm.OLS(train_y, x)
                pvalue = model.fit().pvalues[i + 1]
                # print(pvalue)
                if pvalue < pvalue_min and pvalue < alpha:
                    pvalue_min = pvalue
                    column_min = feature

            if column_min != "":
                feature_col.remove(column_min)
                final_feature.append(column_min)
            else:
                break
        X = sm.add_constant(train_x.loc[:,final_feature])
        model = sm.OLS(train_y, X)
        res = model.fit()
        summary = pd.Series(res.pvalues, index=['const'] + final_feature)
        if ~np.isnan(res.f_pvalue):
            summary['f_test'] = res.f_pvalue
        xx = sm.add_constant(test_x.loc[:,final_feature])
        test_y = res.predict(xx)

    if model_name == 'AdaBoost':
        from sklearn.ensemble import  AdaBoostRegressor
        model = AdaBoostRegressor(n_estimators=100,learning_rate = 0.1)
        adaboost = model.fit(train_x,train_y)
        test_y = pd.Series(adaboost.predict(test_x),index = test_x.index)

    return test_y,summary


# selecting decorator
def selecting(select_step):

    def wrapper(context_dict,f_calendar,*args, **kwargs):
        keys = ['close','volume']
        s = 0
        # trade_status
        for i in keys:
            fd = context_dict[i]
            indexing = f_calendar[-1]
            if s == 0:
                symbols = fd.columns[~fd.loc[indexing, :].isnull()].values
            else:
                flag_nan = ~fd.loc[indexing,:].isnull()
                flag_zero = fd.loc[indexing,:]>0
                temp = fd.columns[flag_zero & flag_nan].values
                symbols = list(set(symbols).intersection(set(temp)))
            s += 1
        # print(len(symbols),' stocks can be traded.')
        return select_step(symbols,context_dict,f_calendar,*args, **kwargs)

    return wrapper

# stoploss strategy
def stoploss(df, re, i, max_new, weight_new):
        flag = 0
        stock = weight_new[weight_new != 0].index.intersection(df.columns)
        # creat indicator for position
        unit = pd.Series(np.full([len(weight_new.index), 1, ], 1)[:, 0], index=weight_new.index)
        unit[weight_new[weight_new < 0].index] *= -1.0
        stop_info = (df[stock].iloc[i - 1] - max_new[stock]) / max_new[stock] * unit[stock]
        if len(stop_info[stop_info < -re]) != 0:
            weight_new[stop_info[stop_info < -re].index] = 0
            weight_new = weight_new / weight_new.sum()
            flag = 1
        return weight_new, flag


