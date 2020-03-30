import datetime
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def price_analysis(Data,buy_tresh,sell_tresh,buysellratio,debug=0):
    trading_cost = 0.005
    buy_index = Data.pct_change() > buy_tresh
    sell_index = Data.pct_change() < sell_tresh
    Euro = 1
    coin = 0
    sellprice = 1000000
    tradingtimes = 0
    actualbuy = []
    actualsell = []
    for index in Data.index:
        if buy_index[index] and (not Euro == 0) and Data[index] < sellprice * buysellratio and index > 2e6:
            buyprice = Data[index]
            coin = Euro * (1 - trading_cost) / buyprice
            Euro = 0
            actualbuy = np.append(actualbuy, index)
            tradingtimes += 1
        if sell_index[index] and (not coin == 0):
            sellprice = Data[index]
            Euro = sellprice * coin * (1 - trading_cost)
            coin = 0
            tradingtimes += 1
            actualsell = np.append(actualsell, index)
    total_assets = Euro + coin * Data.iloc[-1]
    if debug:
        plt.plot(Data.index,Data)
        plt.plot(actualbuy,Data[actualbuy],'*b')
        plt.plot(actualsell,Data[actualsell],'or')
    return total_assets



Data_raw=pd.read_csv('bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')
buycandidate=np.arange(0.01,0.2,0.01)
sellcandidate=np.arange(-0.01,-0.2,-0.01)
buysellcandidate=np.arange(1,1.5,0.02)
timelist=np.arange(12,22,1)
totalsize=buycandidate.size*sellcandidate.size*buysellcandidate.size*timelist.size
assets=[]
i=0
Data_raw = Data_raw.interpolate()
for hour in timelist:

    dt = datetime.datetime(2012, 1, 1, hour)  # starting time of the analysis
    Analyzing_start = dt.timestamp()
    dt = datetime.datetime(2019, 8, 11, hour)
    Analyzing_end = dt.timestamp()
    interval = 60 * 60 * 60  # trading in days
    timeinstances = np.arange(Analyzing_start, Analyzing_end, interval)
    Data = Data_raw[Data_raw.Timestamp.isin(timeinstances)]
    Data = Data.Open
    for buy_tresh in buycandidate:
        for sell_tresh in sellcandidate:
            for buysellratio in buysellcandidate:
                temp= price_analysis(Data,buy_tresh,sell_tresh,buysellratio)
                asset=[hour,buy_tresh,sell_tresh,buysellratio,temp]
                assets.append(asset)
                i+=1
                print(i/totalsize)
    df=pd.DataFrame(assets,columns=['hour','buy_tresh','sell_tresh','buysellratio','totalasset'])
    df.to_pickle('dataframe.pkl')
