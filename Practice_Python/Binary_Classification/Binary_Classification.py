# 분류분석 : Catboost

# library
import pandas as pd
import yfinance as yf
import datetime
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

## score 파일 불러오기
score_dir = "price_data_score_10years.csv"     # score이 담긴 파일(같은 폴더)
score = pd.read_csv(score_dir)
score.columns = ["Today_Date", "Score"]        # Today_Date : 뉴스가 나온 날짜, Score : 모든 뉴스에 대한 개별감성점수

# 모든 날짜 중 개장일과 개장일이 아닌 날을 구별 => 이유 : news는 개장일에만 나오는 것이 아니기 때문에
price_data = yf.download("^DJI",start = '2011-12-30', end = "2022-05-01")   # 2012년 ~ 2022년 4월까지의 다우존스 주가 불러오기
opening = pd.DataFrame(price_data.index.copy())                             # 개장일 추출
opening["opening_date"] = 1                                                 # opening_date : 개장한 날은 1, 그렇지 않은 날은 0

Date = pd.date_range(start='20111230', end='20220430')                      # 모든 날짜 추출
Date = pd.DataFrame({"Date" : Date.values})                                 # 날짜 값을 뽑아와 열이름이 Date인 데이터프레임으로 구성

set_news_data_date = pd.merge(Date, opening, how ="left",left_on='Date', right_on = "Date")  # 모든 날짜 기준으로 개장일을 합침 -> Date와 opening_date 열이 생김
set_news_data_date = set_news_data_date.where(pd.notnull(set_news_data_date), 0)             # 개장되지 않는 날은 opening_date를 0으로 지정(개장일이 아닌 날은 opening의 opening_date에 값이 없으므로 NaN으로 나옴)

# 뉴스가 예측할 날짜를 지정  : ex) 토,일 뉴스 -> 월요일 주가 예측, 월요일 뉴스 -> 화요일 주가 예측  ====>  오늘 뉴스 -> 다음 개장일의 주가 예측
standard=set_news_data_date.Date.iloc[len(set_news_data_date)-1] + datetime.timedelta(days=1)   # 기준 : 2022-05-01
set_news_data_date["predict_date"]=0                                                            # predict_data : 뉴스의 감성점수가 예측하는데 사용될 날짜

for i in range(len(set_news_data_date)-1,-1,-1):                                                # set_news_data_date의 length-1 부터 0까지 -1만큼 생성하기           ** python은 인덱스가 (자리-1)임
    if i==(len(set_news_data_date)-1):                                                                 # 첫번째 루프 즉, i == length-1인 경우 => 데이터프레임의 맨 마지막 값인 경우
        set_news_data_date.predict_date[i]=standard                                                    # 기준을 예측일로 지정 => 기준은 예측할 날짜가 됨
        standard = set_news_data_date.Date[i]                                                          # 기준은 i로 변경 즉, 데이터프레임의 맨 마지막 날짜인 2022-04-30으로 변경
    elif (set_news_data_date.opening_date[i]==1):                                               # 오늘이 개장일인 경우 : 예측일을 기준으로 지정한 후 기준을 오늘로 변경
        set_news_data_date.predict_date[i]=standard                                             # 오늘이 개장일이 아닌 경우 : 기준을 그대로 둠
        standard = set_news_data_date.Date[i]
    else:
        set_news_data_date.predict_date[i]=standard

set_news_data_date.columns = ["Today_Date","Opening_Date","Predict_Date"]                # column 이름 변경

# 뉴스 감성점수 평균 내기
score.Today_Date = score.Today_Date.apply(lambda x: pd.Timestamp(x))                    # str이었던 형식을 time형식으로 변경 => merge를 위해 형식 맞추기
score = pd.merge(set_news_data_date,score, how ="left",on = "Today_Date")               # set_news_data_date를 기준으로 합병 => score에만 있는 Score 열은 set_news_data_date에 날짜가 없는 경우 NaN
score = score[['Today_Date','Predict_Date', "Score"]]                                   # 개장일을 구분했던 Opening_Date를 제외한 나머지를 추출
score = score.groupby('Predict_Date').mean({"Score"})                                   # 예측일을 기준으로 감성점수를 평균 냄 => 즉, 예측일의 주가을 예측하는데 사용될 뉴스의 감성점수를 모두 평균 냄
score = score[score.index<="2022-04-30"]

# 예측날에 맞게 Score과 Today_Close 생성
score = score.reset_index()  # index로 되어있는 Predict_Date를 가져옴
price_data = price_data.reset_index()  # index로 되어있는 Date를 데려옴
price_data = pd.merge(price_data, score, how="left", left_on='Date', right_on="Predict_Date")[
    ["Close", "Score", "Predict_Date"]]  # 예측날의 종가와 예측날에 사용될 Score을 결합
price_data.columns = ["Predict_Close", "Score", "Predict_Date"]  # 변수이름 알맞게 변경

price_data["Today_Close"] = 0  # 예측날을 위해 사용될 오늘의 종가 변수 생성
for i in range(len(price_data) - 1):
    price_data.Today_Close[i + 1] = price_data.Predict_Close[i]  # 예측날은 다음날임을 이용해 값을 생성

first_data = yf.download("^DJI", start='2011-12-29', end='2011-12-30')  # 2012년 1월 2일 전 마지막 개장일 추출
price_data.Today_Close[0] = first_data["Close"][0]  # 2012년 1월 3일의 종가예측을 위해 이전 마지막 개장일이 필요해서 넣음

# price_data = price_data[(price_data["Date"]<"2013-01-01")&("2012-01-01"<=price_data["Date"])]        # 원하는 기간

price_data["Predict_diff"] = 0
price_data["Today_diff"] = 0
for i in range(len(price_data)-1,0,-1):
    price_data.Predict_diff[i] = price_data.Predict_Close[i]-price_data.Predict_Close[i-1]
    if price_data.Predict_diff[i]>=0:
        price_data.Predict_diff[i] = 1
    else: price_data.Predict_diff[i] = 0
    price_data.Today_diff[i] = price_data.Today_Close[i] - price_data.Today_Close[i-1]
    if price_data.Today_diff[i]>=0:
        price_data.Today_diff[i] = 1
    else: price_data.Today_diff[i] = 0

price_data = price_data.iloc[1:]

x = price_data[["Score","Today_Close","Today_diff"]]
y = price_data.Predict_diff

numeric_val = ["Today_Close","Score"]
catagorical_val = ["Predict_diff"]

x = x.reset_index(drop=True)
scaler_x = MinMaxScaler()
x[["Today_Close"]] = pd.DataFrame(scaler_x.fit_transform(x[["Today_Close"]]))

x_train, x_test, y_train, y_test = train_test_split(x, y,shuffle=False)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

###---------------------- CatBoost ---------------------------
n_fold = 15
skfold = TimeSeriesSplit(n_splits=n_fold)
split_ind = []
for train_ind, valid_ind in skfold.split(x_train, y_train):
    split_ind.append((train_ind, valid_ind))

pred = np.zeros((1, len(x_test)))
for i in range(n_fold):
    print("------------ Fold {} ----------".format(i))
    train_ind, valid_ind = split_ind[i]
    X_train = x_train.iloc[train_ind]
    X_val = x_train.iloc[valid_ind]
    Y_train = y_train[train_ind]
    Y_val = y_train[valid_ind]

    catboost = CatBoostClassifier(use_best_model=True, early_stopping_rounds=200, verbose=100)
    train = Pool(data=X_train, label=Y_train)
    val = Pool(data=X_val, label=Y_val)
    catboost.fit(train, eval_set=val, verbose = False)

    catboost_pred = catboost.predict(X_val)
    LL = accuracy_score(Y_val, catboost_pred)
    pred += catboost.predict(x_test)
    print("accuracy : {}".format(LL))

pred = pred[0]/20
for i in range(len(pred)):
    if pred[i]>0.5:
        pred[i]=1
    else: pred[i]=0

print("--------------- CatBoost accuracy : {} --------------------".format(accuracy_score(y_test,pred)))







###---------------------- Random Forest ---------------------------

from sklearn.ensemble import RandomForestClassifier

n_fold = 20
skfold = TimeSeriesSplit(n_splits=n_fold)
split_ind = []
for train_ind, valid_ind in skfold.split(x_train, y_train):
    split_ind.append((train_ind, valid_ind))

pred = np.zeros((1, len(x_test)))
for i in range(n_fold):
    print("------------ Fold {} ----------".format(i))
    train_ind, valid_ind = split_ind[i]
    X_train = x_train.iloc[train_ind]
    X_val = x_train.iloc[valid_ind]
    Y_train = y_train[train_ind]
    Y_val = y_train[valid_ind]

    RF = RandomForestClassifier()
    RF_pred = RF.fit(X_train, Y_train, verbose = False).predict(X_val)
    LL = accuracy_score(Y_val, RF_pred)
    pred += RF.predict(x_test)
    print("accuracy : {}".format(LL))

pred = pred[0]/20
for i in range(len(pred)):
    if pred[i]>0.5:
        pred[i]=1
    else: pred[i]=0

print("--------------- Random Forest accuracy : {} --------------------".format(accuracy_score(y_test,pred)))
