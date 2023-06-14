import pandas as pd
# import pymysql
# from datetime import datetime
# from datetime import timedelta
# import re
# import security
from matplotlib import pyplot as plt
import numpy as np
import KospiAnalyzer as ka
import scipy.optimize as sco
import datetime
from matplotlib import rc
rc('font', family='AppleGothic')
import warnings
warnings.filterwarnings('ignore')



class KospiCAPM:

    def __init__(self, tickers, start_date='2020-01-01', end_date=datetime.datetime.now().strftime('%Y-%m-%d')):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        print(self.start_date)
        print(self.end_date)
        self.rets, self.er, self.cov = self.portfolio(self.start_date, self.end_date)

    def portfolio(self, start_date, end_date):
        # load kospi tickers data
        data = pd.DataFrame()
        for ticker in self.tickers:
            df = ka.KospiTicker().get_daily_price(ticker, start_date, end_date)
            data[ticker] = df['close']
        
        # returns
        rets = data.pct_change().fillna(0)
        # expected returns (yearly)
        er = rets.mean() * 252
        # covariance matrix (yearly)
        cov = rets.cov() * 252

        return rets, er, cov

    # fucntion calculating portfolio statistics
    def statistics(self, weights):

        # weights on portfolio
        weights = np.array(weights)

        # portfolio returns
        pret = np.sum(self.er * weights)

        # portfolio volatility
        pvol = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights)))

        # return ret, vol and sharpe ratio
        return np.array([pret, pvol, pret / pvol])   

    # objective function : portfolio volatility
    def obj_func(self, weights, objective=1):
        return self.statistics(weights)[objective]
    
    # portfolio with maximum sharpe ratio
    def get_msr_weights(self, graph=True):

        # number of assets
        noa = self.er.shape[0]

        # initial equal weights
        init_guess = np.repeat(1 / noa, noa)

        # random weights from random function
        weights = np.random.random(noa)
        weights /= np.sum(weights)

        # weights boundary
        bounds = ((0.0, 1.0),) * noa

        # constraints : imposibility of leverage
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

        er = self.er
        cov = self.cov

        # objective function : negative sharpe ratio
        def neg_sharpe_ratio(weights, er, cov):
            r = weights.T @ er
            # r = np.matmul(np.transpose(weights), self.er)
            vol = np.sqrt(weights.T @ cov @ weights)
            # vol = np.sqrt(np.matmul(np.transpose(weights), self.cov, weights))
            return -r / vol 
        
        # optimization
        res = sco.minimize(neg_sharpe_ratio,
                        init_guess,
                        args=(er, cov),
                        method='SLSQP',
                        constraints=(weights_sum_to_1),
                        bounds=bounds)
        
        
        msr_weights = res.x

        # MSR portfolio returns and volatility
        msr_ret = np.dot(msr_weights, er)
        msr_vol = np.sqrt(np.dot(msr_weights.T, np.dot(cov, msr_weights)))

        print('Return = {}'.format(msr_ret))
        print('Volatility = {}'.format(msr_vol))

        if graph:
            # plot market portfolio
            msr_weights_series = pd.Series(np.round(msr_weights, 4) * 100, index=self.tickers)
            msr_weights_series.plot(kind='bar', title='Market Portfolio', figsize=(12, 8))

        return msr_ret, msr_vol, msr_weights
    

    def get_efficient_frontier(self, simulation=10000, graph=True):

        # # load kospi tickers data
        # data = pd.DataFrame()
        # for ticker in tickers:
        #     df = ka.KospiTicker().get_daily_price(ticker, '2021-01-01')
        #     data[ticker] = df['close']
        
        # # returns
        # rets = data.pct_change().fillna(0)
        # # expected returns (yearly)
        # er = rets.mean() * 252
        # # covariance matrix (yearly)
        # cov = rets.cov() * 252

        rets, er, cov = self.rets, self.er, self.cov

        # empty list for portfolio returns
        p_retunrs = []

        # empty list for portfolio volatility
        p_volatility = []

        # number of assets
        noa = len(self.tickers)

        # number of simulations
        n_ports = simulation

        for i in range(n_ports):

            # random weights
            weights = np.random.random(noa)
            weights /= np.sum(weights)

            # portfolio returns and volatility
            ret = np.dot(weights, er)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

            # append returns and volatilities of each simulation
            p_retunrs.append(ret)
            p_volatility.append(vol)

        # convert to numpy array
        prets = np.array(p_retunrs)
        pvols = np.array(p_volatility)

        # random weights from random function
        weights = np.random.random(noa)
        weights /= np.sum(weights)

        # linspace for portfolio returns
        trets = np.linspace(prets.min(), prets.max(), 100)
        tvols = []

        # volatility level for each return
        for tret in trets:
            # weights : initial equal weights
            init_guess = np.repeat(1 / noa, noa)

            # constraints : return level
            # 'type': 'eq' -> equation
            # 'fun' : lambda x : statistics[x] - tret
            cons = ({'type': 'eq', 'fun': lambda x: self.statistics(x)[0] - tret},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # frotier condition
            bnds = tuple((0.0, 1.0) for x in weights)

            # minimize volatility
            res = sco.minimize(self.obj_func, 
                            init_guess,
                            method='SLSQP',
                            bounds=bnds,
                            constraints=cons)
            
            tvols.append(res['fun'])

        tvols = np.array(tvols)

        # left frontier data
        ind = np.argmin(tvols)
        evols = tvols[ind:]
        erets = trets[ind:]

        # CAPM line
        msr_ret, msr_vol, msr_weights = self.get_msr_weights(graph=True)
        x = np.linspace(evols.min(), evols[-1], 100)
        y = (msr_ret / msr_vol) * x

        if graph:
            # plot efficient frontier
            plt.figure(figsize=(8, 8))
            plt.scatter(x=pvols, y=prets, c=prets / pvols, marker='o')
            plt.plot(evols, erets, 'r', lw=2.0)

            # capital market line
            plt.plot(x, y, 'b', lw=2.0)

            # capital market portfolio
            plt.scatter(msr_vol, msr_ret, marker='*', s=400, color='green')

            plt.grid(True)
            plt.xlabel('Expected volatility')
            plt.ylabel('Expected return')
            plt.title('Portfolio Frontier')


        # return pd.DataFrame({'ind': ind, 'evols': evols, 'erets': erets})
        return msr_ret, msr_vol, msr_weights



