import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split

# !pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
from mpl_finance import candlestick_ohlc
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from sklearn.model_selection import KFold, GridSearchCV
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Conv1D, Concatenate, BatchNormalization, GlobalAveragePooling1D
from keras import Model
from sklearn.preprocessing import OneHotEncoder

price_data = yf.download("AAPL",start = '2016-06-30', end = '2020-06-30')
price_data = price_data[['Open', 'High', 'Low', 'Close', 'Volume']]
price_data = price_data.reset_index()

n = 8
price_data["Future_trend"] = price_data.Close - price_data.Close.shift(n)
price_data.Future_trend[0:(price_data.shape[0]-n)] = price_data.Future_trend[n:(price_data.shape[0])]
price_data = price_data.iloc[0:(price_data.shape[0]-n)]
price_data.Future_trend = price_data.Future_trend.apply(lambda x: 1 if x>0 else 0)
price_data.Date = price_data.Date.apply(lambda x: x.year*10000+x.month*100 + x.day)

train, test = train_test_split(price_data, test_size=0.2, shuffle=False)
train, validation = train_test_split(train, test_size=0.2,shuffle=False)

train_x = np.arange(len(train))
train_ohlc = train[["Date",'Open', 'High', 'Low', 'Close', "Future_trend"]].values
# dohlc = np.hstack((np.reshape(x, (-1, 1)),train_ ohlc))

validation_x = np.arange(len(validation))
validation_ohlc = validation[["Date",'Open', 'High', 'Low', 'Close', "Future_trend"]].values

test_x = np.arange(len(test))
test_ohlc = test[["Date",'Open', 'High', 'Low', 'Close', "Future_trend"]].values


def candlestick_train_save(ohlc):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    candlestick_ohlc(ax, np.array([ohlc[0:5]]), colorup='r', colordown='b')
    plt.axis(False)
    if ohlc[5] == 1:
        plt.savefig('candlestick/train/1/{}.png'.format(int(ohlc[0])))
    else:
        plt.savefig('candlestick/train/0/{}.png'.format(int(ohlc[0])))
    plt.clf()


list(map(candlestick_train_save, train_ohlc))


def candlestick_validation_save(ohlc):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    candlestick_ohlc(ax, np.array([ohlc[0:5]]), colorup='r', colordown='b')
    plt.axis(False)
    if ohlc[5] == 1:
        plt.savefig('candlestick/validation/1/{}.png'.format(int(ohlc[0])))
    else:
        plt.savefig('candlestick/validation/0/{}.png'.format(int(ohlc[0])))
    plt.clf()


list(map(candlestick_validation_save, validation_ohlc))


def candlestick_test_save(ohlc):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    candlestick_ohlc(ax, np.array([ohlc[0:5]]), colorup='r', colordown='b')
    plt.axis(False)
    if ohlc[5] == 1:
        plt.savefig('candlestick/test/1/{}.png'.format(int(ohlc[0])))
    else:
        plt.savefig('candlestick/test/0/{}.png'.format(int(ohlc[0])))
    plt.clf()


list(map(candlestick_test_save, test_ohlc))

image_generator = ImageDataGenerator(rescale = 1/255)

# 바로 아래 코드 실행해서 보면 .ipynb_checkpoints 파일이 함께 잇음
# import os
# train_dir = os.listdir('candlestick/train')
# 위 파일은 자동저장을 하기 위해서 생성된 파일
# 눈으로 보고 삭제할 수 없으므로
# terminal 창에 rm -rf `find -type d -name .ipynb_checkpoints` 입력

train_generator = image_generator.flow_from_directory(
    "candlestick/train",
    target_size=(150,150),
    batch_size = 32,
    class_mode = "binary")

test_generator = image_generator.flow_from_directory(
    "candlestick/test",
    target_size=(150,150),
    batch_size = 32,
    class_mode = "binary")

validation_generator = image_generator.flow_from_directory(
    "candlestick/validation",
    target_size=(150,150),
    batch_size = 32,
    class_mode = "binary")

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2,2))
model.add(Activation('relu'))
model.add(Conv2D(48, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Activation('relu'))
model.add(Conv2D(96, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="softmax"))

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# 최근 버전은 fit_generator을 사용하지 않음
model.fit(train_generator, epochs=100, validation_data= validation_generator, verbose=False)
print(model.evaluate(test_generator))

