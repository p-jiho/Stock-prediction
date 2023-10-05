# Predict stock price of Dow Jones using LSTM for 2012
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

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import backend as K
import matplotlib.pyplot as plt

import datetime
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

from multiprocessing import Pool

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# 데이터 생성, 처리
def time_conversion(time):
    time = pd.Timestamp(time)
    return time


def date_cut_day(dataframe_date_timestamp):
    dataframe_date_timestamp = datetime.datetime.strftime(dataframe_date_timestamp, "%Y-%m-%d")
    return dataframe_date_timestamp


def train_test_result(dir, stock, variable, window_size):
    # score data 뽑기
    score = pd.read_csv(dir)
    score.columns = ["Date", "Score"]

    # price data 뽑기
    price_data = yf.download([stock], start='2012-01-01', end="2022-05-01")

    opening = price_data.index.copy()
    opening = pd.DataFrame(opening)
    opening["opening_date"] = 1
    opening.Date = opening.Date.apply(lambda x: date_cut_day(x))

    Date = pd.date_range(start='20111230', end='20220430')
    Date = pd.DataFrame({"Date": Date.values})
    Date.Date = Date.Date.apply(lambda x: date_cut_day(x))

    set_news_data_date = pd.merge(Date, opening, how="left", left_on='Date', right_on="Date")
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

    set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: date_cut_day(x))

    score = pd.merge(set_news_data_date, score, how="left", on="Date")

    score = score[['Date', 'Score', "price_date"]]
    score = score.groupby('price_date').mean({"Score"})
    score = score[score.index <= "2022-04-29"]

    first_data = yf.download([stock], start='2011-12-30', end='2012-01-01')

    price_data = price_data.reset_index()
    score = score.reset_index()
    score.columns = ["Date", "Score"]

    price_data = price_data[(price_data["Date"] < "2013-01-01") & ("2012-01-01" <= price_data["Date"])]

    # price data와 score data 결합
    price_data = pd.merge(price_data, score, how="left", left_on='Date', right_on="Date")[["Close", "Open", "Score"]]

    # 뉴스가 없는 날은 즉, score가 없는 날은 중립의 의미로 0으로 처리
    price_data.Score[price_data.Score.isnull()] = 0

    # 전날 종가의 영향을 받으므로 전날 종가 변수를 생성
    price_data["before_close"] = 0
    price_data = price_data.reset_index(drop=True)
    for i in range(len(price_data) - 1):
        price_data.before_close[i + 1] = price_data.Close[i]

    price_data.before_close[0] = first_data["Close"][0]

    # minmaxscaler 사용
    price_data.columns = ["Close", "Open", "Score", "before_close"]

    x = price_data[variable]
    y = price_data[["Close"]]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x = pd.DataFrame(scaler_x.fit_transform(x))
    y = pd.DataFrame(scaler_y.fit_transform(y))

    x.columns = variable
    y.columns = ["Close"]

    train_index = int(len(x) * 0.7)

    train_x = x.iloc[0:train_index]
    test_x = x.iloc[train_index:len(x)]

    train_y = y.iloc[0:train_index]
    test_y = y.iloc[train_index:len(y)]

    train_x = train_x.to_numpy().reshape(train_x.shape[0], 1, train_x.shape[1])
    train_y = train_y.to_numpy().reshape(train_y.shape[0], train_y.shape[1])
    test_x = test_x.to_numpy().reshape(test_x.shape[0], 1, test_x.shape[1])
    test_y = test_y.to_numpy().reshape(test_y.shape[0], test_y.shape[1])

    # window size에 맞게 데이터 설정
    x = np.zeros(shape=(train_x.shape[0] - window_size + 1, window_size, x.shape[1]))
    for i in range(train_x.shape[0] - window_size + 1):
        x[i] = np.vstack((train_x[i:i + window_size]))

    y = train_y[window_size - 1:train_x.shape[0]]

    x_t = np.zeros(shape=(test_x.shape[0] - window_size + 1, window_size, x.shape[2]))
    for i in range(test_x.shape[0] - window_size + 1):
        x_t[i] = np.vstack((test_x[i:i + window_size]))

    y_t = test_y[window_size - 1:test_x.shape[0]]

    return scaler_x, scaler_y, x, y, x_t, y_t


dir = "../../../data/Practice_data/price_data_score_10years.csv"
stock = "^DJI"
variable = ["before_close", "Score"]
window_size = 30

scaler_x, scaler_y, x, y, x_t, y_t = train_test_result(dir, stock, variable, window_size)

lstm_1_pred = []
model_input = 0
len_x = x.shape[0]
len_y = y.shape[0]

for i in range(len(x_t)):
    lstm_1_model = Sequential()
    lstm_1_model.add(LSTM(128, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
    lstm_1_model.add(Dropout(0.2))
    lstm_1_model.add((LSTM(64)))
    lstm_1_model.add(Dense(1, activation="tanh"))


    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


    adam = optimizers.Adam(learning_rate=0.0012)
    lstm_1_model.compile(optimizer=adam, loss=root_mean_squared_error, metrics=['mse', "mae"])
    lstm_1_model.fit(x, y, epochs=150, batch_size=64, validation_split=0.2)

    model_input = x_t[i].reshape(1, window_size, len(variable))
    lstm_1_pred = np.append(lstm_1_pred, lstm_1_model.predict(model_input)[0])

    x = np.append(x, x_t[i])
    y = np.append(y, y_t[i])
    len_x += 1
    len_y += 1
    x = x.reshape(len_x, window_size, len(variable))
    y = y.reshape(len_y, 1)

lstm_1_pred = scaler_y.inverse_transform(lstm_1_pred.reshape(len(x_t),1))
with open(file='../../../DeepLearning_result_data/lstm_1_pred_2012.pkl', mode='wb') as f:
    pickle.dump(lstm_1_pred, f)
