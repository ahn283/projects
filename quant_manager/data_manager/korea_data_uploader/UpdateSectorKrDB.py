import json
import requests as rq
import pandas as pd
import keyring
import pymysql
import time
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import numpy as np
import platform

class UpdateSectorKrDB:
    
    def __init__(self, user, pw, host, port, db):
        self.user = user
        self.pw = pw
        self.host = host
        self.port = port
        self.db = db
        self.biz_day = self.get_recent_biz_day()
        self.url = f'http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={self.biz_day}&sec_cd=G10'        
    
    def get_recent_biz_day(self):
        
        # get recent biz day from Naver finance
        url = 'https://finance.naver.com/sise/sise_deposit.nhn'
        data = rq.get(url)
        data_html = BeautifulSoup(data.content)
        parse_day = data_html.select_one('div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah').text
        
        # regex
        biz_day = re.findall('[0-9]+', parse_day)
        biz_day = ''.join(biz_day)
        
        return biz_day
    
    def get_sector_codes(self):
        
        data = rq.get(self.url).json()
        sector_codes = []
        for i in range(len(data['sector'])):
            sector_code = data['sector'][i]['SEC_CD']
            sector_codes.append(sector_code)
        
        return sector_codes
    
    def get_data_from_wics(self):
        sector_codes = self.get_sector_codes()
        
        data_sector = []
        
        for i in tqdm(sector_codes):
            url = f'''http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={self.biz_day}&sec_cd={i}'''
            data = rq.get(url).json()
            data_pd = pd.json_normalize(data['list'])
            
            data_sector.append(data_pd)
            
            time.sleep(2)
            
        # turn data into dataframe        
        kor_sector = pd.concat(data_sector, axis=0)
        
        kor_sector = kor_sector[['IDX_CD', 'CMP_CD', 'CMP_KOR', 'SEC_NM_KOR']]
        kor_sector['date'] = self.biz_day
        kor_sector['date'] = pd.to_datetime(kor_sector['date'])
        
        # rename columns
        kor_sector.rename(columns={
            'IDX_CD': 'index_code', 
            'CMP_CD': 'company_code', 
            'CMP_KOR': 'company', 
            'SEC_NM_KOR': 'sec_nm_kor'
        }, inplace=True)
            
        return kor_sector

    def update_db_sector_kr(self):
        
        # replace NaN into None
        kor_sector = self.get_data_from_wics()
        kor_sector = kor_sector.replace({np.nan: None})
        
        # database info
        # user = 'root'
        # pw = keyring.get_password('macmini_db', user)
        # host = '192.168.219.112' if platform.system() == 'Windows' else '127.0.0.1'
        
        con = pymysql.connect(
            user=self.user,
            passwd=self.pw,
            host=self.host,
            db=self.db,
            charset='utf8'
        )
        
        mycursor = con.cursor()
        query = f"""
        INSERT INTO sector_kr (index_code, company_code, company, sec_nm_kr, date)
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
        index_code=VALUES(index_code), company_code=VALUES(company_code), company=VALUES(company), sec_nm_kr=VALUES(sec_nm_kr)
        """
        
        args = kor_sector.values.tolist()
        
        mycursor.executemany(query, args)
        con.commit()
        
        con.close()      