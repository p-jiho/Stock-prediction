# Stock Market Forecasting Using Machine Learning Algorithms, 2012
# DowJ가 시작되기 전, 발표되는 값들로 당일 종가를 예측한 논문

import yfinance as yf
import pandas as pd

# https://yensr.tistory.com/92
from functools import reduce

start_date = "2011-12-30" # 2012-01-01 이후 첫번째 데이터가 Nan인 경우 선형 보간법을 사용하기 위해 앞의 날짜부터 가져옴
end_date = "2022-04-30"

dow_jones = yf.download('^DJI',start = start_date, end = end_date)
dow_jones = dow_jones.reset_index()[["Date","Close"]]
dow_jones = dow_jones.dropna(axis=0)
dow_jones.columns = ["Dow_Date","DowJ"]

nekkei = yf.download('^N225',start = start_date, end = end_date)
nekkei = nekkei.reset_index()[["Date","Close"]]
nekkei = nekkei.dropna(axis=0)
nekkei.columns = ["Nek_Date","Nekkei"]

hang_seng = yf.download('^HSI',start = start_date, end = end_date)
hang_seng = hang_seng.reset_index()[["Date","Close"]]
hang_seng = hang_seng.dropna(axis=0)
hang_seng.columns = ["HS_Date","HangS"]

FTSE = yf.download('^FTSE',start = start_date, end = end_date)
FTSE = FTSE.reset_index()[["Date","Close"]]
FTSE = FTSE.dropna(axis=0)
FTSE.columns = ["FTSE_Date","FTSE"]

dax = yf.download('^GDAXI',start = start_date, end = end_date)
dax = dax.reset_index()[["Date","Close"]]
dax = dax.dropna(axis=0)
dax.columns = ["DAX_Date","DAX"]

asx = yf.download('^AXJO',start = start_date, end = end_date)
asx = asx.reset_index()[["Date","Close"]]
asx = asx.dropna(axis=0)
asx.columns = ["ASX_Date","ASX"]

eur = yf.download('EURUSD=X',start = start_date, end = end_date)
eur = eur.reset_index()[["Date","Close"]]
eur = eur.dropna(axis=0)
eur.columns = ["EUR_Date","EUR"]

aud = yf.download('AUDUSD=X',start = start_date, end = end_date)
aud = aud.reset_index()[["Date","Close"]]
aud = aud.dropna(axis=0)
aud.columns = ["AUD_Date","AUD"]

jpy = yf.download('JPYUSD=X',start = start_date, end = end_date)
jpy = jpy.reset_index()[["Date","Close"]]
jpy = jpy.dropna(axis=0)
jpy.columns = ["JPY_Date","JPY"]

usd = yf.download('USD',start = start_date, end = end_date)
usd = usd.reset_index()[["Date","Close"]]
usd = usd.dropna(axis=0)
usd.columns = ["USD_Date","USD"]

gold = yf.download('GC=F',start = start_date, end = end_date)
gold = gold.reset_index()[["Date","Close"]]
gold = gold.dropna(axis=0)
gold.columns = ["Gold_Date","Gold"]

silver = yf.download('SI=F',start = start_date, end = end_date)
silver = silver.reset_index()[["Date","Close"]]
silver = silver.dropna(axis=0)
silver.columns = ["Silver_Date","Silver"]

platinum = yf.download('PL=F',start = start_date, end = end_date)
platinum = platinum.reset_index()[["Date","Close"]]
platinum = platinum.dropna(axis=0)
platinum.columns = ["Plati_Date","Platinum"]

oil = yf.download('CL=F',start = start_date, end = end_date)
oil = oil.reset_index()[["Date","Close"]]
oil = oil.dropna(axis=0)
oil.columns = ["Oil_Date","Oil"]


stock_lst = [dow_jones, nekkei, hang_seng, FTSE, dax, asx, eur, aud, jpy, usd, gold, silver, platinum, oil]
stock_df = reduce(lambda x, y: pd.merge(x, y, left_on = x.columns[0], right_on = y.columns[0], how = "left"), stock_lst)

stock_df = stock_df[['Dow_Date', 'DowJ',
       'Nekkei', 'HangS', 'FTSE',
       'DAX', 'ASX', 'EUR',
       'AUD', 'JPY', 'USD', 'Gold',
       'Silver', 'Platinum', 'Oil']]

# 선형보간법
stock_df[stock_df.columns.difference(['Dow_Date'])] = stock_df[stock_df.columns.difference(['Dow_Date'])].interpolate(method='linear',axis=0)
stock_df = stock_df.iloc[1:]
stock_df.to_csv("data/13_variable_close.csv", index = False)