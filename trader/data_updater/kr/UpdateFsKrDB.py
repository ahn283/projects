import pymysql
from sqlalchemy import create_engine
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import time
import keyring
import platform

class UpdateFsKrDB():
    
    def __init__(self):
        
        self.user = 'root'
        self.pw = keyring.get_password('macmini_db', self.user)
        self.host = '192.168.219.106' if platform.system() == 'Windows' else '127.0.0.1'
        self.port = 3306
        self.db = 'stock'
        self.query = """ 
                    INSERT INTO fs_kr (account, date, value, company_code, frequency)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    value=VALUES(value)
                    """
        
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
    
    def clean_fs(df, ticker, frequency):
        
        # clean financial statements
        df = df[~df.loc[:, ~df.columns.isin(['account'])].isna().all(axis=1)]
        df = df.drop_duplicates(['account'], keep='first')
        df = pd.melt(df, id_vars='account', var_name='date', value_name='value')
        df = df[~pd.isnull(df['value'])]
        df['account'] = df['account'].replace({'계산에 참여한 계정 펼치기': ''}, regex=True)
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m') + pd.tseries.offsets.MonthEnd()
        df['ticker'] = ticker
        df['frequency'] = frequency
        
    def update_db_fs_kr(self):
        
        ticker_list = self.read_ticker_list()
        error_list = []
        
        # connect db
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')
        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host=self.host,
            db=self.db,
            charset='utf8'
        )
        mycursor = con.cursor()

        
        # for loop
        for i in tqdm(range(0, len(ticker_list))):
            # select ticker:
            ticker = ticker_list['company_code'][i]
            
            try:
                # url
                url = f'http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A{ticker}'
                
                # get data
                data = pd.read_html(url, displayed_only=False)
                
                # yearly data
                data_fs_y = pd.concat([
                    data[0].iloc[:, ~data[0].columns.str.contains('전년공기')],
                    data[2], data[4]
                ])
                data_fs_y = data_fs_y.rename(columns={data_fs_y.columns[0]: 'account'})
                
                
                # settlement year
                page_data = rq.get(url)
                page_data_html = BeautifulSoup(page_data.content, 'html.parser')
                
                fiscal_data = page_data_html.select('div.corp_group1 > h2')
                fiscal_data_text = fiscal_data[1].text
                fiscal_data_text = re.findall('[0-9]+', fiscal_data_text)
                
                # get only settlement year's data
                data_fs_y = data_fs_y.loc[:, (data_fs_y.columns == 'account') | (data_fs_y.columns.str[-2:].isin(fiscal_data_text))]
                
                # clean data
                data_fs_y_clean = self.clean_fs(data_fs_y, ticker, 'y')
                
                # qaurterly data
                data_fs_q = pd.concat([
                    data[1].iloc[:, ~ data[1].columns.str.contains('전년동기')],
                    data[3], data[5]
                ])
                data_fs_q = data_fs_q.rename(columns={data_fs_q.columns[0]: "account"})
                data_fs_q_clean = self.clean_fs(data_fs_q, ticker, 'q')
                
                # concat yearly and quarterly data
                data_fs_bind = pd.concat([data_fs_y_clean, data_fs_q_clean])
                
                # insert into sb
                args = data_fs_bind.values.tolist()
                mycursor.executemany(query, args)
                con.commit()           
                
                
            except:
                # occuring errors, save the ticker and go to next loop
                print(ticker)
                error_list.append(ticker)
                
            # time sleep
            time.sleep(2)
        
        # db 연결 종료
        engine.dispose()
        con.close()
        