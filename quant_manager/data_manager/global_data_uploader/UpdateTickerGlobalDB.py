from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from datetime import datetime
import math
import pandas as pd 
import numpy as np
from tqdm import tqdm
import time
import pymysql
import keyring
import platform


class UpdateTickerGlobalDB():
    
    def __init__(self, user, pw, host, port, db):
        # # install Chrome driver
        # self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

        # US country code = '5'
        self.country_code = '5'
        self.user = user
        self.pw = pw
        self.host = host
        self.port = port
        self.db = db


        # # first page URL
        # self.url = f'''https://investing.com/stock-screener/?sp=country::
        # {self.country_code}|sector::a|industry::a|equityType::ORD%3Ceq_market_cap;1'''
        
        # # open page with url
        # self.driver.get(url)
        
    def get_ticker_global_data(self):
        
        # install Chrome driver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

        # first page URL
        url = f'''https://investing.com/stock-screener/?sp=country::
        {self.country_code}|sector::a|industry::a|equityType::ORD%3Ceq_market_cap;1'''
        
        # open page with url
        driver.get(url)
        
        # 'Screener Results'shows after table which contains ticker informations is loaded.
        # Wait until table is loaded using WebDriverWait()
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located(
            (By.XPATH, '//*[@id="resultsTable"]/tbody')
        ))
        
        # Crawl number of tickers and count number of pages
        end_num = driver.find_element(By.CLASS_NAME, value='js-total-results').text
        end_num = math.ceil(int(end_num) / 50)
        
        all_data_df = []
        
        for i in tqdm(range(1, end_num + 1)):
            url = f'''https://investing.com/stock-screener/?sp=country::
                {self.country_code}|sector::a|industry::a|equityType::ORD%3Ceq_market_cap;{i}'''
            driver.get(url)
            
            try:
                WebDriverWait(driver, 10).until(EC.visibility_of_element_located(
                    (By.XPATH, '//*[@id="resultsTable"]/tbody')
                ))
                
            except:
                time.sleep(1)
                driver.refresh()
                WebDriverWait(driver, 10).until(EC.visibility_of_element_located(
                    (By.XPATH, '//*[@id="resultsTable"]/tbody')
                ))
                
            html = BeautifulSoup(driver.page_source, 'lxml')
            html_table = html.select(
                'table.genTbl.openTbl.resultsStockScreenerTbl.elpTbl'
            )
            df_table = pd.read_html(html_table[0].prettify())
            df_table_select = df_table[0][['Name', 'Symbol', 'Exchange', 'Sector', 'Market Cap']]
    
            all_data_df.append(df_table_select)
            
            time.sleep(1)
            
        # concatenate all dataframes after ending loop
        all_data_df_bind = pd.concat(all_data_df, axis=0)
        
        data_country = html.find(class_='js-search-input inputDropDown')['value']
        all_data_df_bind['country'] = data_country
        all_data_df_bind['date'] = datetime.today().strftime('%Y-%m-%d')
        
        # delete row whose name is null
        all_data_df_bind = all_data_df_bind[~all_data_df_bind['Name'].isnull()]
        
        
        # select exchange market which can be traded in
        all_data_df_bind = all_data_df_bind[all_data_df_bind['Exchange'].isin(
            ['NASDAQ', 'NYSE', 'NYSE Amex']
        )]
        
        # drop duplicate
        all_data_df_bind = all_data_df_bind.drop_duplicates(['Symbol'])
        all_data_df_bind.reset_index(inplace=True, drop=True)
        all_data_df_bind = all_data_df_bind.replace({np.nan: None})
        
        driver.quit()
        
        return all_data_df_bind
    
    def update_db_ticker_global(self):
        
        all_data_df_bind = self.get_ticker_global_data()

        # database info
        # user = 'root'
        # pw = keyring.get_password('macmini_db', user)
        # host = '192.168.219.106' if platform.system() == 'Windows' else '127.0.0.1'
        user = self.user
        pw = self.pw
        host = self.host
        db = self.db
        
        con = pymysql.connect(
            user=user,
            passwd=pw,
            host=host,
            db=db,
            charset='utf8'
        )
        
        mycursor = con.cursor()
        
        query = """
                INSERT INTO ticker_global (name, symbol, exchange, sector, market_cap, country, date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                name=VALUES(name), exchange=VALUES(exchange), sector=VALUES(exchange), market_cap=VALUES(market_cap);
                """
        
        args = all_data_df_bind.values.tolist()
        
        mycursor.executemany(query, args)
        con.commit()
        
        con.close()
                
