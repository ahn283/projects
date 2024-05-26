from sqlalchemy import create_engine
import pymysql
import pandas as pd
import keyring
from dateutil.relativedelta import relativedelta
import requests as rq
from io import BytesIO
from datetime import date
import re
import time
from tqdm import tqdm
import platform

class UpdatePriceKrDB:
    
    def __init__(self):
        
        self.user = 'root'
        self.pw = keyring.get_password('macmini_db', self.user)
        self.host = '192.168.219.106' if platform.system() == 'Windows' else '127.0.0.1'
        self.port = 3306
        self.db = 'stock'
        self.ticker_list = self.read_ticker_list()
        
    def read_ticker_list(self):
                
        # connect database
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')
        query = """ 
        SELECT * FROM ticker_kr
        WHERE date = (SELECT MAX(date) FROM ticker_kr)
            AND category = '보통주';
        """
        
        # get ticker list from ticker_kr database
        ticker_list = pd.read_sql(query, con=engine)
        engine.dispose()
        
        return ticker_list
    
    def update_db_price_kr(self):
        
        # connect database
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')
        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host=self.host,
            db=self.db,
            charset='utf8'
        )
        mycursor = con.cursor()

        query = """ 
        INSERT INTO price_kr (date, open, high, low, close, volume, company_code)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
        open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close), volume=VALUES(volume);
        """
        
        # list for errors
        error_list = []
        
        # download for all price data (for 50 years)
        for i in tqdm(range(0, len(self.ticker_list))):
            
            # pick a ticker
            ticker = self.ticker_list['company_code'][i]
            
            # start date and end date
            # fr = (date.today() + relativedelta(years=-50)).strftime('%Y%m%d')
            fr = (date.today() + relativedelta(months=-1)).strftime('%Y%m%d')
            to = (date.today()).strftime('%Y%m%d')
            
            # error occurs, skip and do next loop
            try:
                # url : crawling data from Naver
                url = f'''https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={fr}&endTime={to}&timeframe=day'''
                
                # download data
                data = rq.get(url).content
                data_price = pd.read_csv(BytesIO(data))
                
                # data cleaning
                price = data_price.iloc[:, 0:6]
                price.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                price = price.dropna()
                price['date'] = price['date'].str.extract('(\d+)')      # regex : extract only numbers from columns
                price['date'] = pd.to_datetime(price['date'])
                price['company_code'] = ticker
                
                # insert db
                args = price.values.tolist()
                mycursor.executemany(query, args)
                con.commit()
            
            except:
                print(ticker)
                error_list.append(ticker)
                
            # time sleep
            time.sleep(2)
            
        # close db connection
        engine.dispose()
        con.close()
                                                             