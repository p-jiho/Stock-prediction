# 첫번째, 변수에 카테고리를 추가해 카테고리 기준으로 감성분석
import pandas as pd
import copy
import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
dir = "../../../../"
original_news_dir = dir + "data/Practice_data/original_data_4years.pkl"
tit_txt_dir = dir + "data/Practice_data/tit_txt_combination_4years.pkl"

original_news = pd.read_pickle(original_news_dir)
tit_txt = pd.read_pickle(tit_txt_dir)

original_news.date = original_news.date.apply(lambda x: pd.to_datetime(x, errors="ignore")) ## date type으로 변경

## 시간을 날짜 단위로 짜름
def date_until_day(date_timestamp):  ## 2021-01-01 00:23:00 => 2021-01-01
    date_timestamp = datetime.datetime.strftime(date_timestamp, "%Y-%m-%d")
    return date_timestamp


news_day = copy.deepcopy(original_news)     ## original data를 value만 copy
news_day.date = news_day.date.apply(lambda x: date_until_day(x))  ## 시간을 시 단위로 짜름
news_day.reset_index(inplace=True, drop=True)

tit_txt_category= pd.concat([news_day.date,news_day.catagory,tit_txt],axis=1)
tit_txt_category = tit_txt_category.sort_values(by='date')

def del_list_None(lst):
    lst = list(filter(None, lst))
    return lst

tit_txt_category = tit_txt_category.reset_index(drop=True)
tit_txt_category = tit_txt_category.values.tolist()
tit_txt_catagory = list(map(del_list_None, tit_txt_catagory ))

length = -1
tit_txt_combination_day = []

basic_date = 0             ## 기준이 되는 date
for i in range(len(tit_txt_category)):  ## 전체를 한번씩 돈다
    new_date = tit_txt_category[i][0]    # new date는 현재 loop의 date
    if basic_date == new_date:   # 현재 loop의 date가 기준 date와 같으면 실행
        tit_txt_combination_day[length] = tit_txt_combination_day[length]+tit_txt_category[i][1:(len(tit_txt_category[i]))] ## 앞의 데이터에 새로운 데이터를 결합
    else:                        # 현재 loop의 date가 기준 date와 다르면 실행 즉, 새로운 시간이 나타나면 실행
        length += 1             # 길이가 한개 늘어남
        tit_txt_combination_day.append(tit_txt_category[i][1:(len(tit_txt_category[i]))]) # 새로운 list 데이터 추가
        basic_date = tit_txt_category[i][0]              # 기준 date를 새로운 date로 변경

tit_txt_category = pd.DataFrame(tit_txt_category)
tit_txt_token = tit_txt_category.loc[:, 2:tit_txt_category.shape[1]].copy()

def word_to_sentence(lst):  ## 토큰화가 되어있는 tit, tit 데이터를 한 문장으로 만듦
    lst = " ".join(lst)     ## ex) the, and, me, bye => the and me bye
    return lst

## 각 문장마다 수행, 결과 ex) ["the and me bye", ...,"i do not me"]
tit_txt_token = tit_txt_token.values.tolist()                  # dataframe -> list로 변경
tit_txt_token = list(map(del_list_None, tit_txt_token ))       # None 삭제
tit_txt_sentence = list(map(word_to_sentence, tit_txt_token))  ## token을 다시 연결해 문장으로 만듦
tit_txt_sentence = pd.DataFrame(tit_txt_sentence)              # dataframe으로 변경
tit_txt_sentence = pd.concat([tit_txt_category.loc[:,0:1],tit_txt_sentence],axis=1) # sentence에 date를 붙힘
tit_txt_sentence.columns = ["Date","Category","Sentence"]

analyzer = SentimentIntensityAnalyzer()

tit_txt_sentence["Score"]=0

for i in range(len(tit_txt_sentence)):
    score = analyzer.polarity_scores(tit_txt_sentence.Sentence[i])       # 감성점수 추출
    tit_txt_sentence.Score[i] = score["compound"]

score = tit_txt_sentence.groupby(['Date',"Category"]).mean({"Score"})
n = len(set(tit_txt_sentence.Date))
day = pd.to_datetime(list(set(tit_txt_sentence.Date)), errors="ignore").sort_values()
catagory_score = pd.DataFrame()
for i in range(n):
    one_day = str(datetime.datetime.strftime(day[i], "%Y-%m-%d"))
    one_day_score = score.loc[one_day].T.reset_index(drop=True)
    one_day_score.index = [one_day]
    category_score = pd.concat([category_score, one_day_score])

category_score = category_score.where(pd.notnull(category_score), 0)
price_data = yf.download(["^DJI"],start = '2017-01-01', end = "2022-05-01")
close = price_data.Close

Date = pd.date_range(start='20170101', end='20220429')
Date = pd.DataFrame({"Date" : Date.values})
Date.Date = Date.Date.apply(lambda x: date_until_day(x))

category_score = category_score.reset_index()
category_score = pd.merge(Date, category_score, how ="left",left_on='Date', right_on = "index")
category_score = category_score.loc[:,catagory_score.columns!="index"]

prediction_data = pd.DataFrame(close.index)
prediction_data["opening_date"] = 1
prediction_data.Date = prediction_data.Date.apply(lambda x: date_until_day(x))
prediction_data = pd.merge(Date, prediction_data, how ="left",left_on='Date', right_on = "Date")
prediction_data = prediction_data.where(pd.notnull(prediction_data), 0)

category_score = pd.merge(category_score, prediction_data, how ="left",left_on='Date', right_on = "Date")
category_score.Date = category_score.Date.apply(lambda x: pd.to_datetime(x, errors="ignore"))

standard = category_score.Date.iloc[len(category_score) - 1] + datetime.timedelta(days=1)
category_score["price_date"] = 0
for i in range(len(category_score) - 1, -1, -1):
    if i == (len(category_score) - 1):
        standard = category_score.Date[i]
        category_score.price_date[i] = standard + datetime.timedelta(days=1)
    elif (category_score.opening_date[i] == 1) & (category_score.opening_date[i + 1] == 1):
        standard = category_score.Date[i]
        category_score.price_date[i] = standard + datetime.timedelta(days=1)
    elif (i != 0):
        if ((category_score.opening_date[i] == 1) & (category_score.opening_date[i + 1] == 0) & (
                category_score.opening_date[i - 1] == 0)):
            category_score.price_date[i] = standard + datetime.timedelta(days=1)
            standard = category_score.Date[i]
        else:
            category_score.price_date[i] = standard
    else:
        category_score.price_date[i] = standard

category_score.Date = category_score.Date.apply(lambda x: date_until_day(x))
category_score = category_score.dropna(axis=0)
category_score = category_score.groupby('price_date').mean({})
category_score = category_score.loc[:, category_score.columns != "opening_date"]

category_score.to_csv(dir + "data/Practice_data/category_score.csv", index = True)
tit_txt_sentence.to_csv(dir + "data/Practice_data/sentence_and_score.csv", index = False)