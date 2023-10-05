# 두번째, 감성분석
import pandas as pd
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer

import smart_open
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from gensim.models import FastText

from sentence_transformers import SentenceTransformer

import copy

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

dir = "../../../../"
original_news_dir = dir + "data/Practice_data/original_data_4years.pkl"
tit_txt_dir = dir + "data/Practice_data/tit_txt_combination_4years.pkl"

original_news = pd.read_pickle(original_news_dir)
tit_txt_combination = pd.read_pickle(tit_txt_dir)

## date type으로 변경
original_news.date = original_news.date.apply(lambda x: pd.to_datetime(x, errors="ignore"))

## 시간을 날짜 단위로 짜름
def date_until_day(date_timestamp):  ## 2021-01-01 00:23:00 => 2021-01-01
    date_timestamp = datetime.datetime.strftime(date_timestamp, "%Y-%m-%d")
    return date_timestamp


news_day = copy.deepcopy(original_news)     ## original data를 value만 copy
news_day.date = news_day.date.apply(lambda x: date_until_day(x))  ## 시간을 시 단위로 짜름

def del_list_None(lst):
    lst = list(filter(None, lst))
    return lst


## NaN -> None으로 변경(데이터 처리를 편리하게 하기 위해)
tit_txt_combination = tit_txt_combination.where(pd.notnull(tit_txt_combination), None)

tit_txt_combination.reset_index(inplace=True, drop=True)
news_day.reset_index(inplace=True, drop=True)

## 데이터에 date 변수 붙여주기
tit_txt_day = pd.concat([news_day.date,tit_txt_combination],axis=1)


## 시간 순으로 정렬
tit_txt_day = tit_txt_day.sort_values(by='date')

## 새로 정렬된 데이터의 인덱스를 순서대로 재설정  # index 100 51 21 40 ... => 1 2 3 4 5 ...
tit_txt_day = tit_txt_day.reset_index(drop=True)

# 데이터프레임을 list로 변환
tit_txt_day = tit_txt_day.values.tolist()

## 데이터 프레임을 list로 변환하는 과정에서 포함된 None을 삭제
tit_txt_day = list(map(del_list_None, tit_txt_day ))

# 같은 날짜인 경우 데이터를 합침 ex) 1월 1일 00시 00분 ~ 23시 59분 사이의 기사는 전부 합침
length = -1
tit_txt_combination_day = []

basic_date = 0             ## 기준이 되는 date
for i in range(len(tit_txt_day)):  ## 전체를 한번씩 돈다
    new_date = tit_txt_day[i][0]    # new date는 현재 loop의 date
    if basic_date == new_date:   # 현재 loop의 date가 기준 date와 같으면 실행
        tit_txt_combination_day[length] = tit_txt_combination_day[length]+tit_txt_day[i][1:(len(tit_txt_day[i]))] ## 앞의 데이터에 새로운 데이터를 결합
    else:                        # 현재 loop의 date가 기준 date와 다르면 실행 즉, 새로운 시간이 나타나면 실행
        length += 1             # 길이가 한개 늘어남
        tit_txt_combination_day.append(tit_txt_day[i][1:(len(tit_txt_day[i]))]) # 새로운 list 데이터 추가
        basic_date = tit_txt_day[i][0]              # 기준 date를 새로운 date로 변경

tit_txt_day = pd.DataFrame(tit_txt_day)
tit_txt_token = tit_txt_day.loc[:, 1:tit_txt_day.shape[1]].copy()  ## token만 뽑아 오기

def word_to_sentence(lst):  ## 토큰화가 되어있는 tit, tit 데이터를 한 문장으로 만듦
    lst = " ".join(lst)     ## ex) the, and, me, bye => the and me bye
    return lst

## 각 문장마다 수행, 결과 ex) ["the and me bye", ...,"i do not me"]
tit_txt_token = tit_txt_token.values.tolist()                  # dataframe -> list로 변경
tit_txt_token = list(map(del_list_None, tit_txt_token ))       # None 삭제
tit_txt_sentence = list(map(word_to_sentence, tit_txt_token))  ## token을 다시 연결해 문장으로 만듦
tit_txt_sentence = pd.DataFrame(tit_txt_sentence)              # dataframe으로 변경
tit_txt_sentence = pd.concat([tit_txt_day.loc[:,0],tit_txt_sentence],axis=1) # sentence에 date를 붙힘
tit_txt_sentence.columns = ["Date","Sentence"]  # 열이름 새로 지정

analyzer = SentimentIntensityAnalyzer()

tit_txt_sentence["Score"]=0

for i in range(len(tit_txt_sentence)):
    score = analyzer.polarity_scores(tit_txt_sentence.Sentence[i])       # 감성점수 추출
    tit_txt_sentence.Score[i] = score["compound"]

# price data 불러오기
price_data = yf.download(['^DJI'],start = '2016-12-31', end = "2022-05-01")

def date_until_day(date_timestamp):  ## 2021-01-01 00:23:00 => 2021-01-01
    date_timestamp = datetime.datetime.strftime(date_timestamp, "%Y-%m-%d")
    return date_timestamp

## 개장일만 추출
opening = price_data.index.copy()
opening = pd.DataFrame(opening)
opening["opening_date"] = 1

# 개장일 날짜 형태 변경
opening.Date = opening.Date.apply(lambda x: date_until_day(x))

Date = pd.date_range(start='20161230', end='20220430')
Date = pd.DataFrame({"Date" : Date.values})
Date.Date = Date.Date.apply(lambda x: date_until_day(x))

set_news_data_date = pd.merge(Date, opening, how ="left",left_on='Date', right_on = "Date")
set_news_data_date = set_news_data_date.where(pd.notnull(set_news_data_date), 0)         # 개장일과 겹치지 않는 곳은 0
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

'''
standard=set_news_data_date.Date.iloc[len(set_news_data_date)-1] + datetime.timedelta(days=1)
set_news_data_date["price_date"]=0
for i in range(len(set_news_data_date)-1,-1,-1):
    if i==(len(set_news_data_date)-1):
        standard = set_news_data_date.Date[i]
        set_news_data_date.price_date[i]=standard  + datetime.timedelta(days=1)
    elif (set_news_data_date.opening_date[i]==1)&(set_news_data_date.opening_date[i+1]==1):
        standard = set_news_data_date.Date[i]
        set_news_data_date.price_date[i]=standard  + datetime.timedelta(days=1)
    else:
        set_news_data_date.price_date[i]=standard
'''

set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: date_until_day(x))
tit_txt_sentence = tit_txt_sentence[["Date","Score"]]
tit_txt_score = pd.merge(set_news_data_date,tit_txt_sentence, how ="left",on = "Date")
tit_txt_score = tit_txt_score[['Date','Score', "price_date"]]
tit_txt_score = tit_txt_score.groupby('price_date').mean({"Score"})
tit_txt_score = tit_txt_score[tit_txt_score.index<="2022-04-29"]
price_data = price_data["Close"]
price_data_score = pd.concat([price_data,tit_txt_score],axis = 1)
price_data_score = price_data_score[-price_data_score.Score.isnull()]
price_data_score = price_data_score[["Score"]]

price_data_score.to_csv(dir+"data/Practice_data/price_data_score_4years.csv")