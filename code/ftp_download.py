# -*- coding: utf-8 -*-
import urllib.request
import os.path
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import numpy as np
import pandas as pd
path = os.getcwd()
data_path = os.path.join(path + '\\data\\')

def download_data(today):
    # download the dir of ftp file
    web = "ftp://nyu_project:zkGUXJ6atHLH@ftp1.etfg.com/arc_analytics"
    urllib.request.urlretrieve(web,"list.txt")
    # read dir to find the name of list of all data
    f = open("list.txt","r")
    n = f.read()
    rows = n.split("\n")
    file_list = []
    for row in rows:
        word = row.split(" ")
        if ".csv" in word[-1]:
            file_list.append(word[-1])
    # #create the directory to save the downloading file
    if os.path.isdir("analytics"):
        pass
    else:
        os.mkdir("analytics")

    # # download all file to certain directory
    for file in file_list:
        url = web + '/'+file
        loc_file = "analytics/" + file
        urllib.request.urlretrieve(url, loc_file)
        print('.', end='', flush=True)
    print('ftp downloading completed')

    yf.pdr_override()
    #read etf ticks from the first file in order to cover the whole ETFs in the latest data
    first_file = "analytics/" + file_list[0]
    f = pd.read_csv(first_file, header = None)
    col = f[1] # fetch ticker names
    close = pd.DataFrame()
    for tick in col:
        data = pdr.get_data_yahoo(tick,start="2012-02-01", end=today)
        if "Close" in data:
            close[tick] = data["Close"]
        else:
            close[tick] = np.nan

    if os.path.isdir("data"):
        pass
    else:
        os.mkdir("data")
    close.to_csv("data/close.csv")

def close_processing():
    close = pd.read_csv(data_path+'close.csv')
    close.set_index(['Date'], inplace=True)
    ret = (close - close.shift(1)) / close.shift(1)
    close[ret.abs()>2] = np.nan
    close.dropna(how='all', axis=1, inplace=True)
    close.ffill(axis =0,inplace = True)
    close.to_csv("data/close.csv")
    print('preprocessing completed!')

