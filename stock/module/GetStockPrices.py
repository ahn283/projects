import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import module.KospiAnalyzer as KospiAnalyzer

class GetStockPrices:

    # initialize class
    def __init__(self, k_tickers=[], u_tickers=[], param=52, start_date='2010-01-01', end_date=datetime.datetime.today().strftime('%Y-%m-%d'), interval='D'):
        self.ka = KospiAnalyzer.KospiTicker()
        self.data = pd.DataFrame()
        self.data_k = pd.DataFrame()
        self.data_u = pd.DataFrame()
        self.param = param


        self.k_tickers = k_tickers
        self.u_tickers = u_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval


    def get_portfolio_data(self):
        # get korean stock price data
        if len(self.k_tickers) > 0:
            for ticker in self.k_tickers:
                df = self.ka.get_daily_price(ticker, self.start_date, self.end_date)
                self.data_k[ticker] = df['close']

            self.data_k.reset_index(inplace=True)
            self.data_k.date = pd.to_datetime(self.data_k.date).dt.strftime('%Y-%m-%d')
            # print(self.data_k)

        # get us stock price data
        if len(self.u_tickers) > 0:
            self.data_u = yf.download(self.u_tickers, start=self.start_date, end=self.end_date)['Adj Close']
            self.data_u.index = pd.to_datetime(self.data_u.index).strftime('%Y-%m-%d')
            self.data_u.index.name = 'date'

            # get exchange rate data
            usd_krw = pd.DataFrame(yf.download('KRW=X', start=self.start_date, end=self.end_date)['Adj Close'])
            usd_krw.index.name = 'date'
            # self.data_u['usd_krw'] = 1/usd_krw
            self.data_u['usd_krw'] = usd_krw
            print(self.data_u.iloc[:, :-1])

            # convert us stock price to krw
            for index in range(len(self.data_u.columns) - 1):
                # self.data_u.iloc[:, index] = self.data_u.iloc[:, index] * (1/self.data_u.iloc[:, -1])
                self.data_u.iloc[:, index] = self.data_u.iloc[:, index] * (self.data_u.iloc[:, -1])
            print(self.data_u)

        # merge kospi and us stock data
        
        if self.data_k.empty:
            data = self.data_u
        
        elif self.data_u.empty:
            data = self.data_k
        else:
            data = pd.merge(self.data_k, self.data_u, how='outer', on='date')

            data.set_index('date', inplace=True)
            data.sort_index(inplace=True)

        if self.interval == 'M':
            data.index = pd.to_datetime(data.index)
            data = data.resample('M').last()
            print(data)

        return data