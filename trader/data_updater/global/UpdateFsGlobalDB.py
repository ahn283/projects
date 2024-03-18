from sqlalchemy import create_engine
import pymysql
import pandas as pd
from yahooquery import Ticker
import time
from tqdm import tqdm
import numpy as np
import keyring
import platform

class UpdateFsGlobalDB():
    
    def __init__(self):
        self.user = 'root'
        self.pw = keyring.get_password('macmini_db', self.user)
        self.host = '192.168.219.106' if platform.system() == 'Windows' else '127.0.0.1'
        self.port = 3306
        self.db = 'stock'
        
    def get_ticker_list(self):
        
        # connect DB
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')

        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host = self.host,
            db=self.db,
            charset='utf8'
        )
        
        mycursor = con.cursor()
        
        # select ticker list
        ticker_list = pd.read_sql(
            """ 
            SELECT * FROM ticker_global
            WHERE date = (SELECT MAX(date) FROM ticker_global)
            AND country = 'United States';
            """
        , con=engine)
        
        engine.dispose()
        con.close()
        
        return ticker_list
    
    def update_db_fs_global(self):
        
        # connect DB
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')

        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host = self.host,
            db=self.db,
            charset='utf8'
        )
        
        mycursor = con.cursor()
        
        ticker_list = self.get_ticker_list()
        error_list = []
        
        query = """ 
                    INSERT INTO fs_global (ticker, date, account, value, freq)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    value = VALUES(value);
                """
        
        # connect db
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')
        
        # download all fs
        for i in tqdm(range(0, len(ticker_list))):
            # select ticker
            ticker = ticker_list['symbol'][i]
            
            try:
                # download data
                data = Ticker(ticker)
                
                # yearly fs
                data_y = data.all_financial_data(frequency='a')
                data_y.reset_index(inplace=True)
                data_y = data_y.loc[:, ~data_y.columns.isin(['periodType', 'currencyCode'])]
                data_y = data_y.melt(id_vars=['symbol', 'asOfDate'])
                data_y = data_y.replace([np.nan], None)
                data_y['freq'] = 'y'
                data_y.columns = ['ticker', 'date', 'account', 'value', 'freq']
                
                # quarterly fs
                data_q = data.all_financial_data(frequency='q')
                data_q.reset_index(inplace=True)
                data_q = data_q.iloc[:, ~data_q.columns.isin(['periodType', 'currencyCode'])]
                data_q = data_q.melt(id_vars=['symbol', 'asOfDate'])
                data_q = data_q.replace([np.nan], None)
                data_q['freq'] = 'q'
                data_q.columns = ['ticer', 'date', 'account', 'value', 'freq']
                
                # concat
                data_fs = pd.concat([data_y, data_q], axis=0)
                
                # insert into db
                args = data_fs.values.tolist()
                mycursor.executemany(query, args)
                con.commit()
                
            except:
                print(ticker)
                error_list.append(ticker)
                
            time.sleep(1)
            
        # close db connection
        engine.dispose()
        con.close()