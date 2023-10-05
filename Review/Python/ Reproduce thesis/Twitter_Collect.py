import snscrape.modules.twitter as sntwitter
import pandas as pd
from multiprocessing import Pool
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import re
import multiprocessing

from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC as SVC
from sklearn.naive_bayes import GaussianNB as NB
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Conv1D
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling1D

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, GridSearchCV
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop

Date = pd.date_range(start='20160630', end='20200701')
Date = Date.strftime('%Y-%m').unique()
Date = list(Date)


def tweet_collect(Date):
    query = "AAPL"
    start_day = Date + "-01"
    end_day = datetime.strptime(start_day, '%Y-%m-%d') + relativedelta(days=1)
    end_day = end_day.strftime('%Y-%m-%d')

    tweet_list = []
    for tweet in sntwitter.TwitterSearchScraper(query + " since:" + start_day + " until:" + end_day).get_items():
        if (tweet.retweetCount >= 5 and tweet.likeCount >= 10):
            tweet_list.append([tweet.date, tweet.rawContent, tweet.retweetCount, tweet.likeCount])
    return tweet_list


def work_func(Date):
    tweet = list(map(tweet_collect, Date))
    return tweet


def main():
    num_cores = multiprocessing.cpu_count()
    Date_split = np.array_split(Date, num_cores)
    pool = Pool(num_cores)
    tweet = pool.map(work_func, Date_split)
    pool.close()
    pool.join()

    return tweet


if __name__ == "__main__":
    start = datetime.now()
    tweet = main()
    print(datetime.now() - start)

tweets = []
for i in range(len(tweet)):
    for j in range(len(tweet[i])):
        tweets = tweets + tweet[i][j]

tweets_text = [tweets[i][1].replace("\n"," ") for i in range(len(tweets))]
tweets_text = list(map((lambda x: re.sub(r"[^!\"#$%&\'()*+,-./0-9:;<=>?@A-Z[\]^_`a-z{|}~]"," ",x)), tweets_text))


# hyperlink 제거
def remove_link(lst):
    pattern = re.compile("^https://+")
    remove_lst = list(map((lambda x: "" if pattern.search(x) else x), lst.split()))
    remove_lst = list(filter(None, remove_lst))
    remove_lst = ' '.join(remove_lst)
    return remove_lst


tweets_text = list(map(remove_link, tweets_text))

# 구두점 제거
tweets_text = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), tweets_text))
# tokenize
tweets_text = list(map(word_tokenize, tweets_text))


def del_stopword(line):  ## 한 리스트의 한 문장씩 불러와서 dir_stopword_produce 적용
    dir_stop_words = stopwords.words('english')  ## 불용어 사전

    line_stopwords_intersection = list(set(line) & set(dir_stop_words))  ## 각 문장과 불용어 사전에 동시에 있는 단어 추출

    # 각 문장마다 불용어 사전과 교집합인 사전 생성

    # 각 문장마다 교집합 사전에 해당하지 않는 값만 추출
    line = difference(line, line_stopwords_intersection)

    return line


def difference(line, line_stopwords_intersection):  ## 각 문장, 각 문장과 불용어 사전의 교집합 입력
    line = [i for i in line if i not in line_stopwords_intersection]  ## 불용어 사전에 해당하지 않는 단어만 추출
    return line


tweets_text = list(map(del_stopword, tweets_text))
tweets_text = list(map(lambda x: " ".join(x), tweets_text))
sentiment_score = list(map(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)["compound"], tweets_text))
tweets = pd.DataFrame(tweets)
tweets.columns = ["Date", "Text", "Reweet", "Like"]
tweets = tweets.Date
tweets = tweets.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))

sentiment_score = pd.DataFrame(sentiment_score)
sentiment_score = pd.concat([tweets,sentiment_score], axis=1)
sentiment_score.columns = ["Date", "Score"]
sentiment_score.Score = sentiment_score.Score.apply(lambda x: 4 if x>0.5
                           else 3 if 0<x and x <=0.5
                           else 2 if x==0
                           else 1 if -0.5 <= x and x < 0
                           else 0 )
sentiment_score = sentiment_score.groupby('Date').sum({"Score"})
sentiment_score = sentiment_score.reset_index()

price_data = yf.download("AAPL",start = '2016-06-01', end = '2020-07-01')
price_data = price_data[['Open', 'High', 'Low', 'Close', 'Volume']]
price_data = price_data.reset_index()
price_data.Date = price_data.Date.apply(lambda x : x.strftime('%Y-%m-%d'))

input_feature = pd.merge(price_data, sentiment_score, how = "left", on = "Date").dropna()
n = 4
input_feature["Future_trend"] = input_feature.Close - input_feature.Close.shift(n)
input_feature.Future_trend[0:(input_feature.shape[0]-n)] = input_feature.Future_trend[n:(input_feature.shape[0])]
input_feature = input_feature.iloc[0:(input_feature.shape[0]-n)]
input_feature.Future_trend = input_feature.Future_trend.apply(lambda x: 1 if x>0 else 0)
input_feature = input_feature[['Open', 'High', 'Low', 'Close', 'Volume', 'Score', 'Future_trend']]

columns = input_feature.columns
scaler = MinMaxScaler()
input_feature = scaler.fit_transform(input_feature)
input_feature = pd.DataFrame(input_feature)
input_feature.columns = columns

train, test = train_test_split(input_feature, test_size=0.2,shuffle=False)
X_train = train[['Open', 'High', 'Low', 'Close', 'Volume', 'Score']]
Y_train = train[['Future_trend']]
X_test = test[['Open', 'High', 'Low', 'Close', 'Volume', 'Score']]
Y_test = test[['Future_trend']]

RF_model = RF(min_samples_leaf = 1, max_depth = 2, random_state = 0, criterion = "gini")
RF_model.fit(X_train, Y_train)
print("----------------- Random Forest accuracy : {} --------------".format(sum(RF_model.predict(X_test) == Y_test.Future_trend)/Y_test.shape[0]))

SVC_model = SVC(random_state = 42, class_weight = "balanced")
SVC_model.fit(X_train, Y_train)
print("----------------- SVC accuracy : {} --------------".format(sum(SVC_model.predict(X_test) == Y_test.Future_trend)/Y_test.shape[0]))

NB_model = NB().fit(X_train, Y_train)
print("----------------- NB accuracy : {} --------------".format(sum(NB_model.predict(X_test) == Y_test.Future_trend)/Y_test.shape[0]))

train, validation = train_test_split(train, test_size=0.2,shuffle=False)

X_train = train[['Open', 'High', 'Low', 'Close', 'Volume', 'Score']]
Y_train = train[['Future_trend']]

X_validation = validation[['Open', 'High', 'Low', 'Close', 'Volume', 'Score']]
Y_validation = validation[['Future_trend']]
X_test = test[['Open', 'High', 'Low', 'Close', 'Volume', 'Score']]
Y_test = test[['Future_trend']]

ohe = OneHotEncoder(sparse=False)
ohe.fit(Y_train)
Y_train = ohe.transform(Y_train)
Y_validation = ohe.transform(Y_validation)
Y_test = ohe.transform(Y_test)

X_train = X_train.to_numpy().reshape(X_train.shape[0],1,X_train.shape[1])
X_validation = X_validation.to_numpy().reshape(X_validation.shape[0],1,X_validation.shape[1])
X_test = X_test.to_numpy().reshape(X_test.shape[0],1,X_test.shape[1])

model = load_model('model/lstm_model_{}_Dow_Jones.h5'.format(4))
print("----------------- LSTM accuracy : {} --------------".format(model.evaluate(X_test, Y_test)))


lr = 0.001
param_grid={'batch_size' :[8, 16,24],
            'optimizer' : [Adam(lr=lr), SGD(lr=lr), RMSprop(lr=lr)]
           }

def create_cnn(optimizer):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters = 64, kernel_size = 3, input_shape = (1,X_train.shape[2]), padding="same"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('relu'))
    cnn_model.add(Conv1D(filters = 64, kernel_size = 3, padding="same"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('relu'))
    cnn_model.add(Conv1D(filters = 64, kernel_size = 3, padding="same"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('relu'))
    cnn_model.add(GlobalAveragePooling1D())
    cnn_model.add(Dense(32))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(2, activation = "softmax"))
    cnn_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return(cnn_model)

cnn_model = KerasClassifier(build_fn=create_cnn)

Gridcnn = GridSearchCV(estimator=cnn_model,
                     param_grid=param_grid,
                     cv=3)
filename ='copy_2_data/cnn_model_{}_Dow_Jones.h5'.format(n)
checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose=0,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                            )
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20, min_lr=0.0001)
Gridcnn.fit(X_train, Y_train, epochs=100, callbacks=[reduce_lr, checkpoint], validation_data=(X_validation,Y_validation), verbose=0)

model = load_model('model/cnn_model_{}_Dow_Jones.h5'.format(n))
print("----------------- CNN accuracy : {} --------------".format(model.evaluate(X_test, Y_test)))

# RESULT(twitter)
#                 Dow Jonsdes      Apple
#  4 Days  RF                      0.33
#         SVC                      0.28
#         NB                       0.29
#         LSTM                     0.33
#         CNN                    s
