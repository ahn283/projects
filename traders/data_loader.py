# import pymysql
from sqlalchemy import create_engine
import keyring
import platform
import numpy as np
import pandas as pd

# Data columns

COLUMNS_STOCK_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']
COLUMNS_TRAINING_DATA = ['open', 'high', 'low', 'close', 'volume', 'close_ma5', 'volume_ma5', 'close_ma5_ratio', 'volume_ma5_ratio',
       'open_close_ratio', 'open_prev_close_ratio', 'high_close_ratio',
       'low_close_ratio', 'close_prev_close_ratio', 'volume_prev_volume_ratio',
       'close_ma10', 'volume_ma10', 'close_ma10_ratio', 'volume_ma10_ratio',
       'close_ma20', 'volume_ma20', 'close_ma20_ratio', 'volume_ma20_ratio',
       'close_ma60', 'volume_ma60', 'close_ma60_ratio', 'volume_ma60_ratio',
       'close_ma120', 'volume_ma120', 'close_ma120_ratio',
       'volume_ma120_ratio', 'close_ma240', 'volume_ma240',
       'close_ma240_ratio', 'volume_ma240_ratio', 'upper_bb',
       'lower_bb', 'bb_pb', 'bb_width', 'macd',
       'macd_signal', 'macd_oscillator', 'rs', 'rsi']

class DataLoader:
    
    def __init__(self, stock_code='AAPL', start_date='2018-01-01', end_date='2023-12-31'):
        ''' 
        Arguments
        ----------
        - stock_code : unique stock code
        - start_date : start date
        - end_date : end data
        '''
        
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        
        # database connection info
        self.user = 'root'
        self.pw = keyring.get_password('macmini_db', self.user)
        self.host = '192.168.219.106' if platform.system() == 'Windows' else '127.0.0.1'
        self.port = 3306
        self.db = 'stock'
        
        
    def get_stock_data(self):
        
        # connect db
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')

        # get data from the database
        if self.start_date is not None:
            if self.end_date is not None:               
                query = f""" 
                        SELECT * FROM price_global
                        WHERE ticker = '{self.stock_code}'
                        AND date BETWEEN '{self.start_date}' AND '{self.end_date}' 
                        """
            else:
                query = f""" 
                        SELECT * FROM price_global
                        WHERE ticker = '{self.stock_code}'
                        AND date >= '{self.start_date}'
                        """
                        
        else:
            if self.start_date is not None:
                query = f""" 
                        SELECT * FROM price_global
                        WHERE ticker = '{self.stock_code}'
                        AND date <= '{self.start_date}' 
                        """
            else:
                query = f""" 
                        SELECT * FROM price_global
                        WHERE ticker = '{self.stock_code}'
                        """

        print(query)
        stock_data = pd.read_sql(query, con=engine)
        engine.dispose()
        return stock_data[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']]
    
    def preprocess(self, data):

        # moving average
        windows = [5, 10, 20, 60, 120, 240]
        for window in windows:
            data[f'close_ma{window}'] = data['close'].rolling(window).mean()
            data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
            data[f'close_ma{window}_ratio'] = (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
            data[f'volume_ma{window}_ratio'] = (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']
            data['open_close_ratio'] = (data['open'].values - data['close'].values) / data['close'].values
            data['open_prev_close_ratio'] = np.zeros(len(data))
            data.loc[1:, 'open_prev_close_ratio'] = (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
            data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
            data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
            data['close_prev_close_ratio'] = np.zeros(len(data))
            data.loc[1:, 'close_prev_close_ratio'] = (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values 
            data['volume_prev_volume_ratio'] = np.zeros(len(data))
            data.loc[1:, 'volume_prev_volume_ratio'] = (
                # if volume is 0, change it into non zero value exploring previous volume continuously
                (data['volume'][1:].values - data['volume'][:-1].values) / data['volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values
            )
        
        # Bollinger band
        data['middle_bb'] = data['close'].rolling(20).mean()
        data['upper_bb'] = data['middle_bb'] + 2 * data['close'].rolling(20).std()
        data['lower_bb'] = data['middle_bb'] - 2 * data['close'].rolling(20).std()
        data['bb_pb'] = (data['close'] - data['lower_bb']) / (data['upper_bb'] - data['lower_bb'])
        data['bb_width'] = (data['upper_bb'] - data['lower_bb']) / data['middle_bb']
        
        # MACD
        macd_short, macd_long, macd_signal = 12, 26, 9
        data['ema_short'] = data['close'].ewm(macd_short).mean()
        data['ema_long'] = data['close'].ewm(macd_long).mean()
        data['macd'] = data['ema_short'] - data['ema_long']
        data['macd_signal'] = data['macd'].ewm(macd_signal).mean()
        data['macd_oscillator'] = data['macd'] - data['macd_signal']
        
        # RSI
        data['close_change'] = data['close'].diff()
        # data['close_up'] = np.where(data['close_change'] >=0, df['close_change'], 0)
        data['close_up'] = data['close_change'].apply(lambda x: x if x >= 0 else 0)
        # data['close_down'] = np.where(data['close_change'] < 0, df['close_change'].abs(), 0)
        data['close_down'] = data['close_change'].apply(lambda x: -x if x < 0 else 0)
        data['rs'] = data['close_up'].ewm(alpha=1/14, min_periods=14).mean() / data['close_down'].ewm(alpha=1/14, min_periods=14).mean()
        data['rsi'] = 100 - (100 / (1 + data['rs']))
        
        return data


    def load_data(self):
        ''' 
        Arguments
        ----------
        - stock_code : unique stock code
        - fro : start date
        - to : end data
        
        Returns
        --------
        df_adj : entire prerprocessed data
        stock_data : data for plotting chart
        training_data : data for training a model
        '''
        
        df = self.get_stock_data()
        df_adj = self.preprocess(df).dropna().reset_index(drop=True)
        
        stock_data = df_adj[COLUMNS_STOCK_DATA]
        training_data = df_adj[COLUMNS_TRAINING_DATA]
        
        return df_adj, stock_data, training_data.values