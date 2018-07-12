# -*- coding: utf-8 -*-
import urllib.request
import os.path
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import numpy as np
import pandas as pd
import datetime
import warnings
path = os.getcwd()
data_path = os.path.join(path + '\\data\\')
ana_path = os.path.join(path+'\\analytics\\')
warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system.
warnings.filterwarnings('ignore')

def sort_file_list(original_list):
    file_info = dict()
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

    for file in original_list:

        if file[0]!='a':
            loc_a = file.find('a')
            textstr = file[loc_a:file.find('.')]
            datestr = file[:loc_a-1]
            datestr = datetime.datetime.strptime(datestr,'%Y-%m-%d').strftime(format = '%Y%m%d')
        else:
            head_loc = file.find('_') + 1
            tail_loc = file.find('.')
            datestr = file[head_loc:tail_loc]

        if datestr<latest_date:
            continue
        else:
            if datestr not in file_info.keys():
                file_info[datestr]=[file]
            if file[0]!='a':
                file = textstr+'_'+datestr+'.csv'
            file_info[datestr].append(file)

    return file_info

def download_data(today):
    # # download the dir of ftp file
    web = "ftp://nyu_project:zkGUXJ6atHLH@ftp1.etfg.com/analytics"
    # urllib.request.urlretrieve(web,"list.txt")
    # read dir to find the name of list of all data
    f = open("list.txt","r")
    n = f.read()
    rows = n.split("\n")
    file_list = []
    for row in rows:
        word = row.split(" ")
        if ".csv" in word[-1]:
            file_list.append(word[-1])

    file_info = sort_file_list(file_list)
    # # # download all file to certain directory
    for key in file_info.keys():
        url = web + '/'+ file_info[key][0]
        loc_file = "analytics/" + file_info[key][1]
        urllib.request.urlretrieve(url, loc_file)
        print(loc_file,end=',',flush=True)
    print('\n'+'ftp downloading completed')

    yf.pdr_override()
    #read the whole ETFs in the latest data
    exist_list = os.listdir("analytics\\")
    if len(exist_list)>0:
        exist_list.sort(reverse=True)
        first_file = ana_path + exist_list[1]
    else:
        print('analytics downloading failed!')
        return
    f = pd.read_csv(first_file, header = None)
    col = f[1].values# fetch ticker names
    # first get trading_days
    sp500 = pdr.get_data_yahoo('^GSPC',start='2012-02-01',end=today)
    sp500_close = sp500['Adj Close']
    close = pd.DataFrame(index=sp500_close.index)
    volume = pd.DataFrame(index=sp500_close.index)

    for tick in col:
        try:
           data = pdr.get_data_yahoo(tick,start="2012-02-01", end=today)
        except:
           print(tick)
           print('?')
        if "Adj Close" in data:
            temp = data["Adj Close"]
            temp.name = tick
            close = close.join(temp.to_frame(),how='left')
        else:
            close[tick] = np.nan
        if 'Volume' in data:
            temp = data['Volume']
            temp.name = tick
            volume = volume.join(temp,how='left')
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

