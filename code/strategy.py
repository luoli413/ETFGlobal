import pandas as pd
import numpy as np


def __model(model_name, train_x, train_y, test_x,alpha = 0.1,*args,**kwargs):
    summary = None
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    import sklearn
    import statsmodels.regression.linear_model as sm
    if model_name == 'Random':
        test_y = pd.Series(np.random.random_sample((len(test_x),)), index=test_x.index)
    if model_name == 'None':
        test_y = test_x.iloc[:,0]
    if model_name == 'MLPRegressor':
        mlp = MLPRegressor(hidden_layer_sizes=(20, 20))
        mlp.fit(train_x, train_y)
        y_pred = mlp.predict(test_x)
        test_y = pd.Series(y_pred, index=test_x.index)
    if model_name == 'Lasso':
        model = sklearn.linear_model.Lasso(0.001,fit_intercept = False)
        lasso = model.fit(train_x, train_y)
        test_y = pd.Series(lasso.predict(test_x), index=test_x.index)
        summary = lasso.score(train_x,train_y)
    if model_name == 'Ridge':
        model = sklearn.linear_model.Ridge(1.0,fit_intercept = False)
        ridge = model.fit(train_x, train_y)
        test_y = pd.Series(ridge.predict(test_x), index=test_x.index)
        summary = ridge.score(train_x, train_y)
    if model_name == 'SVR':
        svr_rbf = SVR(kernel='rbf', C=1, gamma=0.0001, epsilon=0.1)
        svr_rbf.fit(train_x, train_y)
        y_pred_rbf = svr_rbf.predict(test_x)
        test_y = pd.Series(y_pred_rbf, index=test_x.index)
        summary = svr_rbf.score(train_x,train_y)
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
        if ~np.isnan(res.rsquared_adj):
            summary['score'] = res.rsquared_adj
        xx = sm.add_constant(test_x.loc[:,final_feature],has_constant='raise')
        test_y = res.predict(xx)


    if model_name == 'AdaBoost':
        from sklearn.ensemble import  AdaBoostRegressor
        model = AdaBoostRegressor(n_estimators=100,learning_rate = 0.1)
        adaboost = model.fit(train_x,train_y)
        test_y = pd.Series(adaboost.predict(test_x),index = test_x.index)
        summary = adaboost.score(train_x,train_y)

    if model_name == 'RandomForestRegressor':
        from sklearn.ensemble import RandomForestRegressor
        rfr = RandomForestRegressor(n_estimators=100, criterion='mse',max_features='auto')
        rfr.fit(train_x, train_y)
        y_pred_rfr = rfr.predict(test_x)
        test_y = pd.Series(y_pred_rfr, index=test_x.index)

    return test_y,summary

def models(model_names,*args,**kwargs):

    def combine_method(test_y,score,com_method='mean',*args,**kwargs):
        if com_method=='mean':
            if isinstance(test_y,pd.DataFrame):
                test_y = pd.Series(test_y.mean(axis=1))
        if com_method == 'score':
            score = pd.Series(score,index=list(test_y.columns))

            score[score<0] = 0.0
            summary = score.mean()
            score = score / score.sum()

            if len(score) == test_y.shape[1]:
                test_y = pd.Series(np.dot(np.array(test_y),np.array(score)),\
                                  index=test_y.index)
            else:
                if isinstance(test_y, pd.DataFrame):
                   test_y = pd.Series(test_y.mean(axis=1))

        return test_y,summary

    test_y = pd.DataFrame()
    score = []
    if isinstance(model_names,str):
        model_names = [model_names]
    for model in model_names:
        testy,summary = __model(model,*args,**kwargs)
        if isinstance(testy,pd.Series):
            testy.name = model
        if test_y.empty:
            test_y = testy
        else:
            test_y = test_y.to_frame().join(testy.to_frame(),how='outer')
        score.append(summary)
    test_y,summary = combine_method(test_y,score,**kwargs)

    return test_y,summary

# #             if summary is None:
#             elif isinstance(summary,float):


# Please add models here!!!
# def model(model_name, train_x, train_y, test_x,alpha = 0.1,):
#         summary = None
#         from sklearn.neural_network import MLPRegressor
#         from sklearn.svm import SVR
#         import sklearn
#         import statsmodels.regression.linear_model as sm
#         if model_name == 'Random':
#             test_y = pd.Series(np.random.random_sample((len(test_x),)), index=test_x.index)
#         if model_name == 'None':
#             test_y = test_x.iloc[:,0]
#         if model_name == 'MLPRegressor':
#             mlp = MLPRegressor(hidden_layer_sizes=(20, 20))
#             mlp.fit(train_x, train_y)
#             y_pred = mlp.predict(test_x)
#             test_y = pd.Series(y_pred, index=test_x.index)
#         if model_name == 'Lasso':
#             model = sklearn.linear_model.Lasso(0.001,fit_intercept = False)
#             lasso = model.fit(train_x, train_y)
#             test_y = pd.Series(lasso.predict(test_x), index=test_x.index)
#             summary = lasso.score(train_x,train_y)
#         if model_name == 'Ridge':
#             model = sklearn.linear_model.Ridge(1.0,fit_intercept = False)
#             ridge = model.fit(train_x, train_y)
#             test_y = pd.Series(ridge.predict(test_x), index=test_x.index)
#             summary = ridge.score(train_x, train_y)
#         if model_name == 'SVR':
#             svr_rbf = SVR(kernel='rbf', C=1, gamma=0.0001, epsilon=0.1)
#             svr_rbf.fit(train_x, train_y)
#             y_pred_rbf = svr_rbf.predict(test_x)
#             test_y = pd.Series(y_pred_rbf, index=test_x.index)
#         if model_name == 'StepWise':
#
#             feature_col = list(train_x.columns.values)
#             length = len(feature_col)
#             final_feature = []
#             for i in range(length):
#                 pvalue_min = 1
#                 column_min = ""
#                 for feature in feature_col:
#                     temp_feature = final_feature + [feature]
#                     x = sm.add_constant(train_x.loc[:,temp_feature])
#                     model = sm.OLS(train_y, x)
#                     pvalue = model.fit().pvalues[i + 1]
#                     # print(pvalue)
#                     if pvalue < pvalue_min and pvalue < alpha:
#                         pvalue_min = pvalue
#                         column_min = feature
#
#                 if column_min != "":
#                     feature_col.remove(column_min)
#                     final_feature.append(column_min)
#                 else:
#                     break
#
#             X = sm.add_constant(train_x.loc[:,final_feature])
#             model = sm.OLS(train_y, X)
#             res = model.fit()
#             summary = pd.Series(res.pvalues, index=['const'] + final_feature)
#             if ~np.isnan(res.f_pvalue):
#                 summary['f_test'] = res.f_pvalue
#             if ~np.isnan(res.rsquared_adj):
#                 summary['score'] = res.rsquared_adj
#             xx = sm.add_constant(test_x.loc[:,final_feature],has_constant='raise')
#             test_y = res.predict(xx)
#
#
#         if model_name == 'AdaBoost':
#             from sklearn.ensemble import AdaBoostRegressor
#             model = AdaBoostRegressor(n_estimators=100,learning_rate = 0.5)
#             adaboost = model.fit(train_x,train_y)
#             test_y = pd.Series(adaboost.predict(test_x),index = test_x.index)
#
#         if model_name == 'RandomForestRegressor':
#             from sklearn.ensemble import RandomForestRegressor
#             rfr = RandomForestRegressor(n_estimators=100, criterion='mse',max_features='auto')
#             rfr.fit(train_x, train_y)
#             y_pred_rfr = rfr.predict(test_x)
#             test_y = pd.Series(y_pred_rfr, index=test_x.index)
#
#
#         return test_y,summary
#
#     # def combine_models(self,model_name,*args,**kwargs):


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


