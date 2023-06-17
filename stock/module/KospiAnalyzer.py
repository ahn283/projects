import pandas as pd
import pymysql
from datetime import datetime
from datetime import timedelta
import re
from module import security
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class KospiTicker:
    def __init__(self):
        """생성자: MariaDB 연결 및 종목코드 딕셔너리 생성"""
        self.pw = security.password().db_pw
        self.conn = pymysql.connect(host='localhost', user='root', password=self.pw, db='stock', charset='utf8')
        self.codes = {}
        # get company information from company_info table
        self.get_comp_info()
    
    def __del__(self):
        """소멸자: MariaDB 연결 해제"""
        self.conn.close()
    
    def get_comp_info(self):
        """종목코드에 해당하는 종목명을 반환"""
        sql = 'SELECT * FROM company_info'
        krx = pd.read_sql(sql, self.conn)
        for idx in range(len(krx)):
            self.codes[krx['code'].values[idx]] = krx['company'].values[idx]
    
    def get_daily_price(self, code, start_date=None, end_date=None):
        """KRX 종목별 시세를 데이터프레임 형태로 반환
        - code          : KRS 종목코드('005930') 또는 상장기업명('삼성전자')
        - start_date    : 조회 시작일('2020-01-01'), 미입력 시 1년 전 오늘
        - end_date      : 조회 종료일('2020-12-31'), 미입력 시 오늘 날짜
        """
        if (start_date is None):
            one_year_ago = datetime.today() - timedelta(days=365)
            start_date = one_year_ago.strftime('%Y-%m-%d')
            print("start_date is initialized to '{}'".format(start_date))
        else:
            # using regular expression identiying various date format
            ## \d : digit string equals to [0-9]
            ## \D : non-digit string equals to [^0-9]
            ## \D+ : non-digit string with length 1 or more
            ## only digit string year, month, day returns to start_list
            start_list = re.split('\D+', start_date)
            if (start_list[0] == ''):
                start_list = start_list[1:]
            start_year = int(start_list[0])
            start_month = int(start_list[1])
            start_day = int(start_list[2])
            if (start_year < 1900) or (start_year > 2200):
                print(f"ValueError: start_year({start_year:d}) is wrong.")
                return
            if (start_month < 1) or (start_month > 12):
                print(f"ValueError: start_month({start_month:d}) is wrong.")
                return
            if (start_day < 1) or (start_day > 31):
                print(f"ValueError: start_day({start_day:d}) is wrong.")
                return
            start_date = f'{start_year:04d}-{start_month:02d}-{start_day:02d}'
        
        if (end_date is None):
            end_date = datetime.today().strftime('%Y-%m-%d')
            print("end_date is initialized to '{}'".format(end_date))
        else:
            end_list = re.split('\D+', end_date)
            if (end_list[0] == ''):
                end_list = end_list[1:]
            end_year = int(end_list[0])
            end_month = int(end_list[1])
            end_day = int(end_list[2])
            if (end_year < 1900) or (end_year > 2200):
                print(f"ValueError: end_year({end_year:d}) is wrong.")
                return
            if (end_month < 1) or (end_month > 12):
                print(f"ValueError: end_month({end_month:d}) is wrong.")
                return
            if (end_day < 1) or (end_day > 31):
                print(f"ValueError: end_day({end_day:d}) is wrong.")
                return
            end_date = f'{end_year:04d}-{end_month:02d}-{end_day:02d}'
        
        codes_keys = list(self.codes.keys())
        codes_values = list(self.codes.values())
        if code in codes_keys:
            pass
        elif code in codes_values:
            idx = codes_values.index(code)
            code = codes_keys[idx]
        else:
            print(f"ValueError: Code({code}) doesn't exist.")

        sql = f"SELECT * FROM daily_price WHERE code = '{code}' AND date >= '{start_date}' AND date <= '{end_date}'"
        df = pd.read_sql(sql, self.conn)
        # set index column to date
        df.index = df['date']
        return df
    
    # analyze_stock
    def analyze_stock(self, company, start_date=None, end_date=None, graph=False, rolling_window=20):
        # # feature name creation using company name
        # exec("df_%s = %s" % (company, ka.KospiTicker().get_daily_price('company', start_date, end_date)))
        df = self.get_daily_price(company, start_date, end_date)


        # daily returns
        df['rets_daily'] = df['close'].pct_change().fillna(0)
        # rolling returns (rolling_window = 20 days)
        df['rets_rolling'] = df['close'].pct_change(rolling_window).fillna(0)

        # volatility
        ## rolling volatility (1 year)
        df['vol_yearly'] = df['rets_daily'].rolling(252).std().fillna(0) * np.sqrt(252)
        ## rolling volatility (rolling_window = 20 days)
        df['vol_rolling'] = df['rets_daily'].rolling(rolling_window).std().fillna(0) * np.sqrt(20)

        # Bollinger band
        ## rolling_window = 20 days
        df['MA_rolling'] = df['close'].rolling(window=rolling_window).mean()
        df['MV_rolling'] = df['close'].rolling(window=rolling_window).std()
        df['bolllinger_upper'] = df['MA_rolling'] + (df['MV_rolling'] * 2)
        df['bollinger_lower'] = df['MA_rolling'] - (df['MV_rolling'] * 2)

        ## Bollinger band index: %b 
        df['bollinger_percentage'] = (df['close'] - df['bollinger_lower']) / (df['bolllinger_upper'] - df['bollinger_lower'])

        ## Bollinger band bandwidth
        df['bollinger_bandwidth'] = (df['bolllinger_upper'] - df['bollinger_lower']) / df['MA_rolling'] * 100
        

        # plot graph

        if graph == True:
            plt.figure(figsize=(10, 40))
            # close price graph
            plt.subplot(6, 1, 1)
            plt.plot(df.loc[pd.to_datetime(df.index) > pd.to_datetime(start_date)]['close'])
            plt.title(f'Daily Close Price: {company}')
            # returns graph
            plt.subplot(6, 1, 2)
            plt.plot(df.loc[pd.to_datetime(df.index) > pd.to_datetime(start_date)]['rets_daily'])
            plt.title(f'Daily Returns: {company}')
            # rolling volatility graph
            plt.subplot(6, 1, 3)
            plt.plot(df.loc[pd.to_datetime(df.index) > pd.to_datetime(start_date)]['vol_rolling'])
            plt.title(f'Rolling Volatility: {company}')

            # Bollinger band graph
            plt.subplot(6, 1, 4)
            plt.plot(df.index, df['close'], color='#0000ff', label='Close')
            plt.plot(df.index, df['MA_rolling'], 'k--', label='Moving average 20')
            plt.plot(df.index, df['bolllinger_upper'], 'r--', label='Upper band')
            plt.plot(df.index, df['bollinger_lower'], 'c--', label='Lower band')
            plt.fill_between(df.index, df['bolllinger_upper'], df['bollinger_lower'], color='0.9')
            plt.legend(loc='best')
            plt.title(f'{company} Bollinger Band (20 day, 2 std)')

            # Bollinger band index graph
            plt.subplot(6, 1, 5)
            plt.plot(df.index, df['bollinger_percentage'], 'b', label='%b')
            plt.grid(True)
            plt.legend(loc='best')
            plt.title(f'{company} %B')

            # Bollinger band bandwidth graph
            plt.subplot(6, 1, 6)
            plt.plot(df.index, df['bollinger_bandwidth'], 'm', label='Bandwidth')
            plt.grid(True)
            plt.legend(loc='best')
            plt.title(f'{company} Bandwidth')


        # analyze yesterday's data
        yesterday_data = {'date': df.index[-1], 
                'close': df['close'][-1], 
                'ret_yesterday': df['rets_daily'][-1], 
                'ret_rolling': df['rets_rolling'][-1],
                'vol_yesterday_yearly': df['vol_yearly'][-1], 
                'vol_yesterday_rolling': df['vol_rolling'][-1],
                'bollinger_percentage_yesterday': df['bollinger_percentage'][-1], 
                'bollinger_percentage_rolling_mean': df['bollinger_percentage'][-rolling_window:].mean(),
                'bollinger_bandwidth_yesterday': df['bollinger_bandwidth'][-1], 
                'bollinger_bandwidth_rolling_mean': df['bollinger_bandwidth'][-rolling_window:].mean()}
        df_analyzed = pd.DataFrame(yesterday_data, index=[0])
        df_analyzed.set_index('date', inplace=True)
        
        print('-----------------')
        print(f'{company} yesterday data')
        print('return yesterday: ', df_analyzed['ret_yesterday'][0])
        print('volatiltity yesterday: ', df_analyzed['vol_yesterday_rolling'][0])
        print('Bollinger band percentage yesterday: ', df_analyzed['bollinger_percentage_yesterday'][0])
        print('Bollinger band percentage 20 days mean: ', df_analyzed['bollinger_percentage_rolling_mean'][0])
        print('Bollinger band bandwidth: ', df_analyzed['bollinger_bandwidth_yesterday'][0])
        print('Bollinger band bandwidth 20 days mean: ', df_analyzed['bollinger_bandwidth_rolling_mean'][0])
        return df_analyzed
