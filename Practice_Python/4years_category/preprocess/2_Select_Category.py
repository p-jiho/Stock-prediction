# 두번째, 카테고리를 포함한 감성분석 결과를 이용해 후진선택법으로 유의한 카테고리만 추출

import pandas as pd
from statsmodels.formula.api import ols
import yfinance as yf
import datetime
import statsmodels.api as sm

dir = "../../../../"
category_score = pd.read_csv(dir + "data/Practice_data/category_score.csv")

def date_until_day(date_timestamp):  ## 2021-01-01 00:23:00 => 2021-01-01
    date_timestamp = datetime.datetime.strftime(date_timestamp, "%Y-%m-%d")
    return date_timestamp

price_data = yf.download(["^DJI"],start = '2017-01-01', end = "2022-05-01")
price_data = price_data.Close

opening = price_data.index.copy()
opening = pd.DataFrame(opening)
opening["opening_date"] = 1
opening.Date = opening.Date.apply(lambda x: date_until_day(x))

price_data = price_data.reset_index(drop=False)
price_data.Date = price_data.Date.apply(lambda x: date_until_day(x))
price_data = pd.merge(category_score.price_date,price_data, how='left',left_on = "price_date", right_on = "Date")
price_data.dropna(axis=0)
price_data = price_data.Close

category_score = category_score.iloc[0:1323]
category_score = category_score.loc[:, category_score.columns != 'price_date']

y = price_data
x = category_score

variable = category_score.columns.tolist()

select_variable = category_score.columns.tolist()
standard_pvalue = 0.05

step = 0

while len(variable) > 0:
    select_x = x[select_variable]
    model = sm.OLS(y, select_x).fit()
    max_pvalue = model.pvalues[model.pvalues == max(model.pvalues)]

    select_variable.remove(max_pvalue.index[0])

    if max_pvalue.values[0] < standard_pvalue:
        break

sentence_score = pd.read_csv(dir + "sentence_and_score.csv")
sentence_score = sentence_score.loc[sentence_score["Category"].isin(select_variable)]

Date = pd.date_range(start='20161230', end='20220430')
Date = pd.DataFrame({"Date" : Date.values})
Date.Date = Date.Date.apply(lambda x: date_until_day(x))

set_news_data_date = pd.merge(Date, opening, how ="left",left_on='Date', right_on = "Date")
set_news_data_date = set_news_data_date.where(pd.notnull(set_news_data_date), 0)
set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: pd.to_datetime(x, errors="ignore"))

standard = set_news_data_date.Date.iloc[len(set_news_data_date) - 1] + datetime.timedelta(days=1)
set_news_data_date["price_date"] = 0
for i in range(len(set_news_data_date) - 1, -1, -1):
    if i == (len(set_news_data_date) - 1):
        standard = set_news_data_date.Date[i]
        set_news_data_date.price_date[i] = standard + datetime.timedelta(days=1)
    elif (set_news_data_date.opening_date[i] == 1) & (set_news_data_date.opening_date[i + 1] == 1):
        standard = set_news_data_date.Date[i]
        set_news_data_date.price_date[i] = standard + datetime.timedelta(days=1)
    elif (i != 0):
        if ((set_news_data_date.opening_date[i] == 1) & (set_news_data_date.opening_date[i + 1] == 0) & (
                set_news_data_date.opening_date[i - 1] == 0)):
            set_news_data_date.price_date[i] = standard + datetime.timedelta(days=1)
            standard = set_news_data_date.Date[i]
        else:
            set_news_data_date.price_date[i] = standard
    else:
        set_news_data_date.price_date[i] = standard

set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: date_until_day(x))

category_score = pd.merge(set_news_data_date,sentence_score, how ="left",on = "Date")
category_score = category_score[["price_date",'Score']]
category_score = category_score.groupby('price_date').mean({"Score"})
category_score = category_score[category_score.index<="2022-04-29"]

category_score.to_csv(dir + "data/Practice_data/category_selected_score.csv", index= True)