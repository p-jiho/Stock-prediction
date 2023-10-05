# Stock Market Forecasting Using Machine Learning Algorithms, 2012
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))+"/10year_data/review")

import train_test as tt

import yfinance as yf
from sklearn.svm import SVC

# https://yensr.tistory.com/92
from functools import reduce


dir = "../../../data/Practice_data/price_data_score_10years.csv"
stock = "^DJI"
variable = ["Score"]
window_size = 1
start_date = "2011-12-30"
end_date = "2022-05-03"


scaler_x, scaler_y,x, y, x_t, y_t = tt.train_test_result(dir, stock, variable, window_size, start_date, end_date)

start_date = "2011-12-30"
end_date = "2022-05-04"

dow_jones = yf.download('^DJI',start = start_date, end = end_date)
dow_jones = dow_jones.reset_index()[["Date","Close"]]
dow_jones = dow_jones.dropna(axis=0)
dow_jones.columns = ["Dow_Date","DowJ"]

nasdaq = yf.download('^IXIC',start = start_date, end = end_date)
nasdaq = nasdaq.reset_index()[["Date","Close"]]
nasdaq = nasdaq.dropna(axis=0)
nasdaq.columns = ["NAS_Date","NASDAQ"]


sp500 = yf.download('^GSPC',start = start_date, end = end_date)
sp500 = sp500.reset_index()[["Date","Close"]]
sp500 = sp500.dropna(axis=0)
sp500.columns = ["SP_Date","SP500"]

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

stock_lst = [dow_jones, nasdaq, sp500, nekkei, hang_seng, FTSE, dax, asx, eur, aud, jpy, usd, gold, silver, platinum, oil]
stock_df = reduce(lambda x, y: pd.merge(x, y, left_on = x.columns[0], right_on = y.columns[0], how = "left"), stock_lst)

stock_df = stock_df[['Dow_Date', 'DowJ', 'NASDAQ', 'SP500',
       'Nekkei', 'HangS', 'FTSE',
       'DAX', 'ASX', 'EUR',
       'AUD', 'JPY', 'USD', 'Gold',
       'Silver', 'Platinum', 'Oil']]

# 선형보간법
stock_df[stock_df.columns.difference(['Dow_Date'])] = stock_df[stock_df.columns.difference(['Dow_Date'])].interpolate(method='linear',axis=0)

stock_df = stock_df.drop(["Dow_Date"], axis = 1)

difference = 1
stock_df = stock_df.apply(lambda x: (x - x.shift(difference))/x.shift(difference), axis = 0)
stock_df = stock_df.apply(lambda x: x/abs(x), axis = 0)
stock_df = stock_df.iloc[1:len(stock_df)]
stock_df = stock_df.fillna(0)

score = x.reshape(len(x),).tolist() + x_t.reshape(len(x_t),).tolist()
stock_df["Score"] = score[1:len(score)]

train_ind = int(len(stock_df)*0.7)
train = stock_df.iloc[0:train_ind]
test = stock_df.iloc[train_ind:len(stock_df)]

def svm_accuracy(train, test, x_n, y_n, kern):
    svc = SVC(kernel = kern).fit(train[[x_n]], train[[y_n]])
    pre = svc.predict(test[[x_n]])
    accuracy = sum(pre == test[y_n])/len(test)
    return accuracy

x_n = stock_df.columns.drop(["DowJ", "NASDAQ","SP500"])
y_n = "DowJ"
kern = "rbf"

acc_result = list(map(lambda x: svm_accuracy(train, test, x, y_n, kern), x_n))
acc_result = pd.DataFrame([acc_result], columns = x_n)

# case 1. 14개의 변수 모두 사용
svc_14 = SVC(kernel = kern).fit(train[x_n], train[[y_n]])
svc_14_acc = sum(svc_14.predict(test[x_n]) == test[y_n])/len(test)
print("----------- Case 1 accuracy : {} ----------".format(svc_14_acc))


# case 2. 상위 4개의 변수 모두 사용
acc_result = acc_result.transpose()
acc_result.columns = ["acc"]
x_n_4 = acc_result.sort_values(by="acc").tail(4).index.tolist()

svc_4 = SVC(kernel = kern).fit(train[x_n_4], train[[y_n]])
svc_4_acc = sum(svc_4.predict(test[x_n_4]) == test[y_n])/len(test)
print("----------- Case 2 accuracy : {} ----------".format(svc_4_acc))

# case 3. 상위 6개의 변수 모두 사용
acc_result = acc_result.transpose()
acc_result.columns = ["acc"]
acc_result.sort_values(by="acc")
x_n_4 = acc_result.sort_values(by="acc").tail(4).index.tolist()

svc_4 = SVC(kernel = kern).fit(train[x_n_4], train[[y_n]])
svc_4_acc = sum(svc_4.predict(test[x_n_4]) == test[y_n])/len(test)
print("----------- Case 3 accuracy : {} ----------".format(svc_14_acc))

################## CatBoost #################################
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.fit(train[x_n], train[[y_n]], verbose = False)
preds_class = model.predict(test[x_n])

print("------------ CatBoost accuracy : {} -----------".format(sum((np.array(list(itertools.chain(*preds_class))) == test[y_n].reset_index(drop=True)).values)/len(test[y_n])))


################## Random Forest ########################
stock_lst = [dow_jones, nasdaq, sp500, nekkei, hang_seng, FTSE, dax, asx, eur, aud, jpy, usd, gold, silver, platinum, oil]
stock_df = reduce(lambda x, y: pd.merge(x, y, left_on = x.columns[0], right_on = y.columns[0], how = "left"), stock_lst)

stock_df = stock_df[['Dow_Date', 'DowJ', 'NASDAQ', 'SP500',
       'Nekkei', 'HangS', 'FTSE',
       'DAX', 'ASX', 'EUR',
       'AUD', 'JPY', 'USD', 'Gold',
       'Silver', 'Platinum', 'Oil']]

# 선형보간법
stock_df[stock_df.columns.difference(['Dow_Date'])] = stock_df[stock_df.columns.difference(['Dow_Date'])].interpolate(method='linear',axis=0)

stock_df = stock_df.drop(["Dow_Date"], axis = 1)

difference = 1
stock_df = stock_df.apply(lambda x: (x - x.shift(difference))/x.shift(difference), axis = 0)
stock_df = stock_df.apply(lambda x: x/abs(x), axis = 0)
stock_df = stock_df.iloc[1:len(stock_df)]
stock_df = stock_df.fillna(0)

stock_df[['DowJ', 'NASDAQ', 'SP500',
       'Nekkei', 'HangS', 'FTSE',
       'DAX', 'ASX']] = stock_df[['DowJ', 'NASDAQ', 'SP500',
       'Nekkei', 'HangS', 'FTSE',
       'DAX', 'ASX']].shift(1)
stock_df = stock_df.dropna()

train_ind = int(len(stock_df)*0.7)
train = stock_df.iloc[0:train_ind]
test = stock_df.iloc[train_ind:len(stock_df)]

def svm_accuracy(train, test, x_n, y_n, kern):
    svc = SVC(kernel = kern).fit(train[x_n], train[[y_n]])
    pre = svc.predict(test[x_n])
    accuracy = sum(pre == test[y_n])/len(test)
    return accuracy

x_n = stock_df.columns.drop(["DowJ", "NASDAQ","SP500"])
y_n = "DowJ"
kern = "rbf"

print("-------------- SVM accuracy : {} -------------".format(svm_accuracy(train, test, x_n, y_n, kern))