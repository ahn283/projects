import yfinance as yf
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import time
from tqdm import tqdm
import keyring
import platform

class UpdatePriceGlobalDB():
    def __init__(self, user='root', pw='', host='192.168.219.112', port=3306, db='stock'):
        self.user = user
        self.pw = pw
        self.host = host
        self.port = port
        self.db = db
        
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
        
        mycurosr = con.cursor()
        
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
        
    def update_db_price_global(self):
        
        ticker_list = self.get_ticker_list()
        error_list = []
        
        # connect db
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')

        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host = self.host,
            db=self.db,
            charset='utf8'
        )
        
        mycurosr = con.cursor()
        
        query = """ 
                INSERT INTO price_global (date, high, low, open, close, adj_close, volume, ticker)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                high=VALUE(high), low=VALUES(low), open=VALUES(open), close=VALUES(close), volume=VALUES(volume), adj_close=VALUES(adj_close);
                """
        
        # download all prices of all tickers
        for i in tqdm(range(0, len(ticker_list))):
            
            # select a ticker
            ticker = ticker_list['symbol'][i]
            
            # if error happens, it will be ignored
            try:
                # download price
                price = yf.download(ticker, progress=False)
                
                # clean data
                price = price.reset_index()
                price['ticker'] = ticker
                
                # insert into DB
                args = price.values.tolist()
                mycurosr.executemany(query, args)
                con.commit()
            
            except:
                print(ticker)
                error_list.append(ticker)
                
            time.sleep(1)
            
        # close the db connection
        engine.dispose()
        con.close()
        
        
