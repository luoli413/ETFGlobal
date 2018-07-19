from env_context import context
import pandas as pd
import numpy as np


if __name__=="__main__":

    ben = None
    variable_list = ['quant_technical_st', 'quant_technical_it', 'quant_technical_lt', 'quant_sentiment_iv', \
                     'quant_sentiment_si', 'quant_sentiment_pc', 'quant_fundamental_pe', 'quant_fundamental_pb', \
                     'quant_fundamental_pcf', 'quant_fundamental_div', 'quant_global_sector', 'quant_global_country', \
                     'quant_quality_diversification', 'quant_quality_firm', 'quant_quality_liquidity','quant_grade']
    start_day ='2012-01-01'
    horizon = 21
    method = 'IC'
    context = context(ben,start_day,variable_list)
    context.generate_train(horizon,True,True,method='98%shrink')
    context.feature_selection(horizon,method)
