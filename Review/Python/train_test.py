#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import regex as re
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# 데이터 생성, 처리
def time_conversion(time):
    time = pd.Timestamp(time)
    return time

def date_cut_day(dataframe_date_timestamp):
    dataframe_date_timestamp = datetime.datetime.strftime(dataframe_date_timestamp, "%Y-%m-%d")
    return dataframe_date_timestamp

def train_test_result(dir, stock, variable, window_size, start_date, end_date):
    #score data 뽑기
    score = pd.read_csv(dir)
    score.columns = ["Date", "Score"]
    
    end_1_next_date = pd.to_datetime(end_date, errors="ignore") + datetime.timedelta(days=1)
    end_1_next_date = end_1_next_date.strftime('%Y-%m-%d')
    
    #price data 뽑기
    price_data = yf.download([stock],start = start_date, end = end_1_next_date)
    
    opening = price_data.index.copy()
    opening = pd.DataFrame(opening)
    opening["opening_date"] = 1
    opening.Date = opening.Date.apply(lambda x: date_cut_day(x))
    
    start_2_previous_date = pd.to_datetime(start_date, errors="ignore") - datetime.timedelta(days=2)
    start_2_previous_date = start_2_previous_date.strftime('%Y%m%d')
    
    Date = pd.date_range(start=start_2_previous_date, end=pd.to_datetime(end_date).strftime('%Y%m%d'))
    Date = pd.DataFrame({"Date" : Date.values})
    Date.Date = Date.Date.apply(lambda x: date_cut_day(x))

    
    set_news_data_date = pd.merge(Date, opening, how ="left",left_on='Date', right_on = "Date") 
    set_news_data_date = set_news_data_date.where(pd.notnull(set_news_data_date), 0) 
    set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: pd.to_datetime(x, errors="ignore"))

    standard=set_news_data_date.Date.iloc[len(set_news_data_date)-1] + datetime.timedelta(days=1)
    set_news_data_date["price_date"]=0
    for i in range(len(set_news_data_date)-1,-1,-1):
        if i==(len(set_news_data_date)-1):
            standard = set_news_data_date.Date[i]
            set_news_data_date.price_date[i]=standard  + datetime.timedelta(days=1)
        elif (set_news_data_date.opening_date[i]==1)&(set_news_data_date.opening_date[i+1]==1):
            standard = set_news_data_date.Date[i]
            set_news_data_date.price_date[i]=standard  + datetime.timedelta(days=1)
        elif (i!=0):
            if((set_news_data_date.opening_date[i]==1)&(set_news_data_date.opening_date[i+1]==0)&(set_news_data_date.opening_date[i-1]==0)):
                set_news_data_date.price_date[i]=standard  + datetime.timedelta(days=1)
                standard = set_news_data_date.Date[i]
            else:
                set_news_data_date.price_date[i]=standard
        else:
            set_news_data_date.price_date[i]=standard
    
    set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: date_cut_day(x))
    
    score = pd.merge(set_news_data_date,score, how ="left",on = "Date")
    
    score = score[['Date','Score', "price_date"]]
    score = score.groupby('price_date').mean({"Score"})
    end_1_previous_date = pd.to_datetime(end_date, errors="ignore") - datetime.timedelta(days=1)
    end_1_previous_date = end_1_previous_date.strftime('%Y-%m-%d')
    score = score[score.index<=end_1_previous_date]

    start_2_previous_date = pd.to_datetime(start_date, errors="ignore") - datetime.timedelta(days=2)
    start_2_previous_date = start_2_previous_date.strftime('%Y-%m-%d')
    first_data = yf.download([stock],start = start_2_previous_date, end = start_date)
    
    price_data = price_data.reset_index()
    score = score.reset_index()
    score.columns = ["Date", "Score"]
    
    #price data와 score data 결합
    price_data = pd.merge(price_data, score, how ="left",left_on='Date', right_on = "Date")[["Close","Open","Score"]]

    #뉴스가 없는 날은 즉, score가 없는 날은 중립의 의미로 0으로 처리
    price_data.Score[price_data.Score.isnull()]=0

    #전날 종가의 영향을 받으므로 전날 종가 변수를 생성
    price_data["before_close"] = 0
    price_data = price_data.reset_index(drop=True)
    for i in range(len(price_data)-1):
        price_data.before_close[i+1] = price_data.Close[i]
    
    price_data.before_close[0] = first_data["Close"][0]

    # minmaxscaler 사용
    price_data.columns = ["Close","Open","Score","before_close"]
    
    x = price_data[variable]
    y = price_data[["Close"]]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x = pd.DataFrame(scaler_x.fit_transform(x))
    y = pd.DataFrame(scaler_y.fit_transform(y))
    
    x.columns = variable
    y.columns = ["Close"]

    train_index = int(len(x)*0.7)

    train_x = x.iloc[0:train_index]
    test_x = x.iloc[train_index:len(x)]

    train_y = y.iloc[0:train_index]
    test_y = y.iloc[train_index:len(y)]

    train_x = train_x.to_numpy().reshape(train_x.shape[0],1,train_x.shape[1])
    train_y = train_y.to_numpy().reshape(train_y.shape[0],train_y.shape[1])
    test_x = test_x.to_numpy().reshape(test_x.shape[0],1,test_x.shape[1])
    test_y = test_y.to_numpy().reshape(test_y.shape[0],test_y.shape[1])
    
    #window size에 맞게 데이터 설정
    x = np.zeros(shape=(train_x.shape[0]-window_size+1,window_size,x.shape[1]))
    for i in range(train_x.shape[0]-window_size+1):
        x[i]=np.vstack((train_x[i:i+window_size]))

    y = train_y[window_size-1:train_x.shape[0]]

    x_t = np.zeros(shape=(test_x.shape[0]-window_size+1,window_size,x.shape[2]))
    for i in range(test_x.shape[0]-window_size+1):
        x_t[i]=np.vstack((test_x[i:i+window_size]))
    
    y_t = test_y[window_size-1:test_x.shape[0]]
    
    return scaler_x, scaler_y, x, y, x_t, y_t

