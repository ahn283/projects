import pymysql
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import keyring
import platform

class UpdateValueKrDB():
    
    def __init__(self, user, pw, host, port, db):
        
        # self.user = 'root'
        # self.pw = keyring.get_password('macmini_db', self.user)
        # self.host = '192.168.219.112' if platform.system() == 'Windows' else '127.0.0.1'
        # self.port = 3306
        # self.db = 'stock'
        self.user = user
        self.pw = pw
        self.host = host
        self.port = port
        self.db = db
    
    def get_data(self):
        
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')
        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host=self.host,
            db=self.db,
            charset='utf8'   
        )
        mycursor = con.cursor()
        
        # quarterly financial statements
        kor_fs = pd.read_sql(
        """
        SELECT * FROM fs_kr
        WHERE frequency='q'
        AND account IN ('당기순이익', '자본', '영업활동으로인한현금흐름', '매출액');
        """
        , con=engine)
        
                # ticker list
        ticker_list = pd.read_sql(
            """ 
            SELECT * FROM ticker_kr
            WHERE date = (SELECT MAX(date) FROM ticker_kr)
            AND category = '보통주';
            """
        , con=engine)
        
        engine.dispose()
        con.close()
        
        # TTM calcaulation
        kor_fs = kor_fs.sort_values(['company_code', 'account', 'date'])
        kor_fs['ttm'] = kor_fs.groupby(['company_code', 'account'], as_index=False)['value'].rolling(
            window=4, min_periods=4
        ).sum()['value']
        
        # replace capital as average value
        kor_fs['ttm'] = np.where(kor_fs['account'] == '자본', kor_fs['ttm'] / 4, kor_fs['ttm'])
        kor_fs = kor_fs.groupby(['account', 'company_code']).tail(1)
        
        # add market cap
        kor_fs_merge = kor_fs[['account', 'company_code', 'ttm']].merge(
            ticker_list[['company_code', 'market_cap', 'date']],
            on='company_code'
        )
        
        kor_fs_merge['market_cap'] = kor_fs_merge['market_cap'] / 100000000
        
        return ticker_list, kor_fs_merge
    
    def cal_value_ratio(self, data):
        data['value'] = data['market_cap'] / data['ttm']
        data['ratio'] = np.where(
            data['account'] == '매출액', 'PSR',
            np.where(
                data['account'] == '영업활동으로인한현금흐름', 'PCR',
                np.where(data['account'] == '자본', 'PBR',
                         np.where(data['account'] == '당기순이익', 'PER', None))
            )
        )
        
        data = data[['company_code', 'date', 'ratio', 'value']]
        data = data.replace([np.inf, -np.inf, np.nan], None)
        
        return data
    
    def upate_value_kr_db(self):
        
        query = """ 
        INSERT INTO value_kr (company_code, date, ratio, value)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        value = VALUES(value)
        """
        
        ticker_list, kor_fs = self.get_data()
        kor_fs_merge = self.cal_value_ratio(kor_fs)
        
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.pw}@{self.host}:{self.port}/{self.db}')
        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host=self.host,
            db=self.db,
            charset='utf8'   
        )
        mycursor = con.cursor()
        
        args_fs = kor_fs_merge.values.tolist()
        mycursor.executemany(query, args_fs)
        con.commit()
        
        # dividend yield (DY)
        ticker_list['value'] = ticker_list['dividend'] / ticker_list['close']
        ticker_list['value'] = ticker_list['value'].round(4)
        ticker_list['ratio'] = 'DY'
        dy_list = ticker_list[['company_code', 'date', 'ratio', 'value']]
        dy_list = dy_list.replace([np.inf, -np.inf, np.nan], None)
        dy_list = dy_list[dy_list['value'] != 0]
        
        # insert dy into databases
        args_dy = dy_list.values.tolist()
        mycursor.executemany(query, args_dy)
        con.commit()
        
        engine.dispose()
        con.close()