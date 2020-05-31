from binance.client import Client
from datetime import datetime
import pandas as pd
import numpy as np
import mykeys
import logging
from scipy import signal
coins=['BTC','ETH']
client = Client(mykeys.api_key, mykeys.api_secret)
def get_last_price(coin):
    klines = client.get_historical_klines( coin+'USDT', '1h', '1000h ago UTC')
    col = ['open time', 'open', 'high', 'low', 'close', 'volume', 'colsetime', 'qoteAsetVolume', 'ntrade']
    Data = pd.DataFrame(klines)
    df = Data.iloc[:, :-3]
    df.columns = col
    sig=df['close'].astype(float)
    return sig

def sellorder(coin):
    balance= client.get_asset_balance(coin)
    order = client.create_order(
    symbol=coin+'USDT',
    side=Client.SIDE_SELL,
    type=Client.ORDER_TYPE_MARKET,
    quantity=balance['free'])

def buyorder(coin):
    cash= float(client.get_asset_balance('USDT')['free'])
    avg_price = float(client.get_avg_price(symbol=coin+'USDT')['price'])
    order = client.create_order(
    symbol=coin+'USDT',
    side=Client.SIDE_BUY,
    type=Client.ORDER_TYPE_MARKET,
    quantity=cash/len(coins)/avg_price*(1-0.0015))

def get_dic():
    coins=pd.read_csv('trading_frequency.csv',index_col='coins')
    coins=coins['frequency']
    dic=coins.to_dict()
    return dic

def get_last_trade(coin):
    orders = client.get_all_orders(symbol=coin+'USDT', limit=1)
    last_price=float(orders[-1]['cummulativeQuoteQty'])/float(orders[-1]['executedQty'])
    return last_price, orders[-1]['side']



def decision(sig,f,quantity,last_trade_price,buy_selltresh=0.001):
    numerator_coeffs, denominator_coeffs = signal.butter(2, f)
    filtered = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    if (filtered[-2] > filtered[-1]) and (filtered[-2] > filtered[-3]) and quantity>0.01 and abs(filtered[-1]-last_trade_price)/last_trade_price>buy_selltresh:
        decision_out='sell'
    elif (filtered[-2] < filtered[-1]) and (filtered[-2] < filtered[-3]) and quantity<0.01 and abs(filtered[-1]-last_trade_price)/last_trade_price>buy_selltresh:
        decision_out='buy'
    else:
        decision_out='NA'
    return decision_out

dic=get_dic()
for coin in coins:
    sig=get_last_price(coin)
    f=dic[coin]
    cash = float(client.get_asset_balance(asset='USDT')['free'])
    quantity = float(client.get_asset_balance(asset=coin)['free'])
    last_price, type = get_last_trade('BTC')
    decision_out=decision(sig, f, quantity, last_price, buy_selltresh=0.001)
    if decision_out =='buy':
        buyorder(coin)
    elif dicision_out == 'sell':
        sellorder(coin)



# balance = client.get_asset_balance(asset='BTC')
# klines=client.get_historical_klines('BTCUSDT','1h','1000h ago UTC')
# col=['open time','open','high','low','close','volume','colsetime','qoteAsetVolume','ntrade']
# Data=pd.DataFrame(test)
# df = Data.iloc[:,:-3]
# df.columns=col
# # order = client.create_order(
# # symbol='BTCUSDT',
# # side=Client.SIDE_SELL,
# # type=Client.ORDER_TYPE_MARKET,
# # quantity=balance['free'])
#
# Data=pd.DataFrame(klines)
# df = Data.iloc[:,:-3]
# filename=datetime.now().strftime("%Y%m%d%H%M%S")+'.pkl'
# col=['open time','open','high','low','close','volume','colsetime','qoteAsetVolume','ntrade']
# df.columns=col
# df.to_pickle(filename)