# -*- coding: utf-8 -*-
import urllib.request
import os.path
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import numpy as np
import pandas as pd
import datetime
path = os.getcwd()
data_path = os.path.join(path + '\\data\\')
ana_path = os.path.join(path+'\\analytics\\')

def download_data(today):
    # download the dir of ftp file
    web = "ftp://nyu_project:zkGUXJ6atHLH@ftp1.etfg.com/analytics"
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
        exist_list = os.listdir("analytics\\")
        exist_list.sort()
        if len(exist_list)>=2:
            latest_file = exist_list[-2]
            head_loc = latest_file.find('_')+1
            tail_loc = latest_file.find('.')
            latest_date = latest_file[head_loc:tail_loc]

    else:
        os.mkdir("analytics")

    # # download all file to certain directory
    for file in file_list:
        url = web + '/'+file

        if file[0]!='a':
            loc_a = file.find('a')
            textstr = file[loc_a:file.find('.')]
            datestr = file[:loc_a-1]
            datestr = datetime.datetime.strptime(datestr,'%Y-%m-%d').strftime(format = '%Y%m%d')
        else:
            head_loc = file.find('_') + 1
            tail_loc = file.find('.')
            datestr = file[head_loc:tail_loc]

        if datestr<=latest_date:
            continue
        else:
            if file[0]!='a':
                file = textstr+'_'+datestr+'.csv'

        loc_file = "analytics/" + file
        urllib.request.urlretrieve(url, loc_file)
        print(file,end=',',flush=True)
    print('\n'+'ftp downloading completed')

    yf.pdr_override()
    #read etf ticks from the first file in order to cover the whole ETFs in the latest data
    file = file_list[0]
    if file[0]!='a':
        loc_a = file.find('a')
        textstr = file[loc_a:file.find('.')]
        datestr = file[:loc_a-1]
        file = textstr+'_'+\
               datetime.datetime.strptime(datestr,'%Y-%m-%d').strftime(format = '%Y%m%d')+'.csv'

    first_file = ana_path + file
    f = pd.read_csv(first_file, header = None)
    col = f[1].values# fetch ticker names
    close = pd.DataFrame()
    volume = pd.DataFrame()
    for tick in col:
        try:
           data = pdr.get_data_yahoo(tick,start="2012-02-01", end=today)
        except:
           print(tick)
           print('?')
        if "Adj Close" in data:
            close[tick] = data["Adj Close"]
        else:
            close[tick] = np.nan
        if 'Volume' in data:
            volume[tick] = data["Volume"]
        else:
            volume[tick] = np.nan

    if os.path.isdir("data"):
        pass
    else:
        os.mkdir("data")
    close.to_csv(data_path+"close.csv")
    volume.to_csv(data_path+"volume.csv")

def data_processing():
    close = pd.read_csv(data_path+'close.csv')
    close.set_index(['Date'], inplace=True)
    ret = (close - close.shift(1)) / close.shift(1)
    close[ret.abs()>2] = np.nan
    close.dropna(how='all', axis=1, inplace=True)
    close.ffill(axis =0,inplace = True)
    close.to_csv(data_path+'close.csv')

    volume = pd.read_csv(data_path+'volume.csv')
    volume.set_index(['Date'], inplace=True)
    volume.dropna(how='all', axis=1, inplace=True)
    volume.to_csv(data_path+'volume.csv')
    print('preprocessing completed!')

