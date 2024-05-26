import requests as rq
from bs4 import BeautifulSoup
import re
from io import BytesIO
import pandas as pd
import numpy as np
import keyring
import pymysql
import platform

class UpdateTickerKrDB:
    
    def __init__(self):
        self.biz_day = self.get_recent_biz_day()
        self.gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
        self.mktID = {
            'KOSPI' : 'STK',    # KOSPI ID
            'KOSDAQ' : 'KSQ'    # KOSDAQ ID
        }
        self.biz_day = self.get_recent_biz_day()
        
        # add a referrer in the header
        # we can get OTP from the first url, when sending this to the second url without a referrer, web site recognizes this request as one from a bot.
        self.headers = {'Referer':  'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader'}
        
    def get_recent_biz_day(self):
        
        # get recent biz day from Naver finance
        url = 'https://finance.naver.com/sise/sise_deposit.nhn'
        data = rq.get(url)
        data_html = BeautifulSoup(data.content)
        parse_day = data_html.select_one('div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah').text
        
        # regex for converting date into 'yyyymmdd' format
        biz_day = re.findall('[0-9]+', parse_day)
        biz_day = ''.join(biz_day)
        
        return biz_day
    
    def gen_otp_krx(self, market='KOSPI'):
        
        if market == 'Ind':
            # generate OTP for individual stock info
            gen_otp = {
                'searchType': '1',
                'mktId': 'ALL',
                'trdDd': self.biz_day,
                'csvxls_isNo': 'false',
                'name': 'fileDown',
                'url': 'dbms/MDC/STAT/standard/MDCSTAT03501'
            }
        else:
            # generate OTP for KOSPI/KOSDAQ market info
            gen_otp = {
                'mktId' : self.mktID[market],        # STK는 코스피
                'trdDd' : self.biz_day,
                'money' : '1',
                'csvxls_isNo' : 'false',
                'name' : 'fileDown',
                'url' : 'dbms/MDC/STAT/standard/MDCSTAT03901'
            }
        
        # send queries by post() funciton, get data and select only text info.
        otp_stk = rq.post(self.gen_otp_url, gen_otp, headers=self.headers).text

        print(f"generating OTP mktID: {gen_otp['mktId']}")
        
        return otp_stk
    
    def download_data_krx(self, market='KOSPI'):
        
        # generate OTP
        otp = self.gen_otp_krx(market)
        down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
        down_sector = rq.post(down_url, {'code': otp}, headers=self.headers)
        
        sector = pd.read_csv(BytesIO(down_sector.content), encoding='EUC-KR')
        
        return sector
    
    def get_combined_data_krx(self):
        
        # download KOSPI, KOSDAQ and individual stock info
        sector_stk = self.download_data_krx(market='KOSPI')
        sector_ksq = self.download_data_krx(market='KOSDAQ')
        krx_sector = pd.concat([sector_stk, sector_ksq]).reset_index(drop=True)
        
        # delete blank in the company_name
        krx_sector['종목명'] = krx_sector['종목명'].str.strip()
        
        # add 'date' column for recent business day
        krx_sector['date'] = self.biz_day
        
        # get individual stock data from KRX
        krx_ind = self.download_data_krx(market='Ind')
        krx_ind['종목명'] = krx_ind['종목명'].str.strip()
        krx_ind['data'] = self.biz_day
        
        # get difference list between krx_sector and krx_ind
        diff = list(set(krx_sector['종목명']).symmetric_difference(set(krx_ind['종목명'])))
        
        # diff codes are not normal, so we just merge two data bases
        kor_ticker = pd.merge(
            krx_sector,
            krx_ind,
            on = krx_sector.columns.intersection(
                krx_ind.columns
            ).to_list(), how='outer'
        )
        
        # distinguish general stock from SPAC, preferred stock, REITs, other stocks
        kor_ticker['category'] = np.where(kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'), '스팩',
                                          np.where(kor_ticker['종목코드'].str[-1:] != '0', '우선주',
                                                   np.where(kor_ticker['종목명'].str.endswith('리츠'), '리츠',
                                                            np.where(kor_ticker['종목명'].isin(diff), '기타', '보통주'))))
        
        kor_ticker = kor_ticker.reset_index(drop=True)
        kor_ticker.columns = kor_ticker.columns.str.replace(' ', '')    # delete blank from column names
        kor_ticker = kor_ticker[['종목코드', '종목명', '시장구분', '종가',
                                 '시가총액', 'date', 'EPS', '선행EPS', 'BPS', '주당배당금', 'category']]
        kor_ticker['date'] = self.biz_day
        
        # rename columns name into english
        kor_ticker.rename(columns={
            '종목코드': 'company_code',
            '종목명': 'company',
            '시장구분': 'market',
            '종가': 'close',
            '시가총액': 'market_cap', 
            'EPS': 'eps', 
            '선행EPS': 'forward_eps', 
            'BPS': 'bps', 
            '주당배당금': 'dividend'
        }, inplace=True)
        
        return kor_ticker
    
    def update_db_ticker_kr(self, kor_ticker):
        
        # replace NaN into None
        kor_ticker = kor_ticker.replace({np.nan: None})
        
        # database info
        user = 'root'
        pw = keyring.get_password('macmini_db', user)
        host = '192.168.219.106' if platform.system() == 'Windows' else '127.0.0.1'
        
        con = pymysql.connect(
            user=user,
            passwd=pw,
            host=host,
            db='stock',
            charset='utf8'
        )
        
        mycursor = con.cursor()
        query = f"""
        INSERT INTO ticker_kr (company_code, company, market, close, market_cap, date, eps, forward_eps, bps, dividend, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        company=VALUES(company), market=VALUES(market), close=VALUES(close), market_cap=VALUES(market_cap), eps=VALUES(eps),
        forward_eps=VALUES(forward_eps), bps=VALUES(bps), dividend=VALUES(dividend), category=VALUES(category);
        """
        
        args = kor_ticker.values.tolist()
        
        mycursor.executemany(query, args)
        con.commit()
        
        con.close()
        
