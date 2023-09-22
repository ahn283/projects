import pymysql
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import urllib
import requests
import time, calendar, json
from threading import Timer
import security

class DBUpdater:
    def __init__(self):
        
        """생성자: MariaDB 연결 및 종목코드 딕셔너리 생성"""

        # read Database password
        self.pw = security.password().db_pw


        # connect to MariaDB
        self.conn = pymysql.connect(
            host='localhost',
            user='root',
            password=self.pw,
            db='stock',
            charset='utf8'
        )

        # create cursor
        ## create company_info and daily_price table if not exists
        with self.conn.cursor() as curs:
            sql = """
            CREATE TABLE IF NOT EXISTS oecd_index (
                location VARCHAR(20),
                indicator VARCHAR(20),
                subject VARCHAR(20),
                measure VARCHAR(20),
                frequency VARCHAR(20),
                time DATE,
                value FLOAT,
                flag VARCHAR(20)
            )
            """

            curs.execute(sql)
            # curs.close()
        
        # commit
        self.conn.commit()

        self.codes = dict()

    
    def update_oecd_index(self, file_path):
        df = pd.read_csv(file_path)
        df.rename(columns={'LOCATION':'location', 'INDICATOR':'indicator', 'SUBJECT':'subject', 'MEASURE':'measure', 'FREQUENCY':'frequency', 'TIME':'time', 'Value':'value', 'Flag Codes':'flag'}, inplace=True)
        with self.conn.cursor() as curs:
            for r in df.itertuples():
                sql = f"REPLACE INTO oecd_index VALUES ('{r.location}', '{r.indicator}', '{r.subject}', '{r.measure}', '{r.frequency}', '{r.time}', {r.value}, '{r.flag}')"
                print(sql)
                curs.execute(sql)
                # curs.close()
            self.conn.commit()
            print('OECD index DB update complete.')


if __name__ == '__main__':
    dbu = DBUpdater()
    cli_file_path = './data/oecd_cli.csv'
    bci_file_path = './data/oecd_bci.csv'
    cci_file_path = './data/oecd_cci.csv'
    dbu.update_oecd_index(cli_file_path)
    dbu.update_oecd_index(bci_file_path)
    dbu.update_oecd_index(cci_file_path)