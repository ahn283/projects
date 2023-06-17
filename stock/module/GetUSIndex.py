import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import sqlite3
import os
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


    
def get_data(ticker_code, index_name, peroiod="max"):
    # get the data
    data = yf.Ticker(ticker_code)
    data = data.history(period=peroiod)
    data.insert(0, "code", ticker_code)
    data.insert(0, "index_name", index_name)

    ## change column name into small letters
    data = data.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Dividends": "dividends", "Stock Splits": "stock_splits"})
    data.index.name = "date"

    ## change data format
    data.index = pd.to_datetime(data.index, format='%d-%m-%Y')
    data.index = data.index.strftime('%Y-%m-%d')
    data["open"] = data["open"].astype(float)
    data["high"] = data["high"].astype(float)
    data["low"] = data["low"].astype(float)
    data["close"] = data["close"].astype(float)
    data["volume"] = data["volume"].astype(int)
    data["dividends"] = data["dividends"].astype(float)
    data["stock_splits"] = data["stock_splits"].astype(float)

    ## mange missing data
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)
    data = data.round(2)

    return data
    
# us stock indexes code: dow jones, s&p 500, nasdaq, russell 2000
# us stock sector indexes code: information technology, health care, consumer discretionary, consumer staples, energy, financials, industrials, materials, real estate, utilities
# us stock sector indexes code: XLK, XLV, XLY, XLP, XLE, XLF, XLI, XLB, XLRE, XLU

def get_index_data(codes=["^DJI", "^GSPC", "^IXIC", "^RUT", "XLK", "XLV", "XLY", "XLP", "XLE", "XLF", "XLI", "XLB", "XLRE", "XLU"],
                    indexes=["dow_jones", "s&p_500", "nasdaq", "russell_2000", "information_technology", "health_care", "consumer_discretionary", "consumer_staples", "energy", "financials", "industrials", "materials", "real_estate", "utilities"]):
        
    index_data_df = pd.DataFrame()

    for code, index in zip(codes, indexes):

        print("start = {}, {}".format(code, index))

        index_data = get_data(code, index)
        index_data_df = pd.concat([index_data_df, index_data])

        print("end = {}, {}".format(code, index))
            

    print(index_data_df.head())
    print(index_data_df.tail())
    print(index_data_df.shape)

    return index_data_df
