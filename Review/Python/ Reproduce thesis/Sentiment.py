import pandas as pd
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

from keras.models import load_model

score = pd.read_csv("../../../../data/Practice_data/price_data_score_10years.csv")
sentiment_score = score.copy()
sentiment_score.Score = sentiment_score.Score.apply(lambda x: 4 if x>0.5
                           else 3 if 0<x and x <=0.5
                           else 2 if x==0
                           else 1 if -0.5 <= x and x < 0
                           else 0 )

sentiment_score = sentiment_score.groupby('Date').sum({"Score"})
sentiment_score = sentiment_score.reset_index()

price_data = yf.download("^DJI",start = '2011-12-31', end = '2022-05-01')
price_data = price_data[['Open', 'High', 'Low', 'Close', 'Volume']]
price_data = price_data.reset_index()
price_data.Date = price_data.Date.apply(lambda x : x.strftime('%Y-%m-%d'))

input_feature = pd.merge(price_data, sentiment_score, how = "left", on = "Date")
n = 6
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
print("--------------- Random Forest accuracy : {} -------------".format(sum(RF_model.predict(X_test) == Y_test.Future_trend)/Y_test.shape[0]))

SVC_model = SVC(random_state = 42, class_weight = "balanced")
SVC_model.fit(X_train, Y_train)
print("--------------- SVC accuracy : {} -------------".format(sum(SVC_model.predict(X_test) == Y_test.Future_trend)/Y_test.shape[0]))

NB_model = NB().fit(X_train, Y_train)
print("--------------- NB accuracy : {} -------------".format(sum(NB_model.predict(X_test) == Y_test.Future_trend)/Y_test.shape[0]))


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

# from sklearn.model_selection import KFold, GridSearchCV
lr = 0.001
param_grid={'batch_size' :[8, 16,24],
            'optimizer' : [Adam(lr=lr), SGD(lr=lr), RMSprop(lr=lr)]
           }

def create_lstm(optimizer):
    lstm_model = Sequential()
    lstm_model.add(LSTM(30, input_shape = (1,X_train.shape[2]),return_sequences=True))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(256))
    lstm_model.add(Dense(2, activation = "softmax"))
    lstm_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print("{}".format(optimizer))
    return(lstm_model)

lstm_model = KerasClassifier(build_fn=create_lstm)

GridLSTM = GridSearchCV(estimator=lstm_model,
                     param_grid=param_grid,
                     cv=3)
filename ='model/lstm_model_{}_Dow_Jones.h5'.format(n)
checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
                             monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
                             verbose=0,            # 로그를 출력합니다
                             save_best_only=True,  # 가장 best 값만 저장합니다
                             mode='auto'           # auto는 알아서 best를 찾습니다. min/max
                            )
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001)
GridLSTM.fit(X_train, Y_train, epochs=100, callbacks=[reduce_lr, checkpoint], validation_data=(X_validation,Y_validation), verbose=0)

model = load_model('model/lstm_model_{}_Dow_Jones.h5'.format(8))
print("----------------- LSTM accuracy : {} ------------------".format(model.evaluate(X_test, Y_test)))


# from sklearn.model_selection import KFold, GridSearchCV

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

model = load_model('copy_2_data/cnn_model_{}_Dow_Jones.h5'.format(n))
print("----------------- CNN accuracy : {} ------------------".format(model.evaluate(X_test, Y_test)))

# RESULT(news)
#                 Dow Jonsdes      Apple
#  4 Days  RF                      0.48
#         SVC                     *0.56
#         NB                      *0.56
#         LSTM                    *0.56
#         CNN                     *0.56
# ------------------------------------------
#  6 Days  RF         0.45         0.59
#         SVC         0.44         0.59
#         NB          0.60         0.59
#         LSTM        0.61         0.59
#         CNN         0.43         0.59
# ------------------------------------------
#  8 Days  RF         0.48         0.47
#         SVC         0.46         0.45
#         NB         *0.60        *0.59
#         LSTM        0.59        *0.59
#         CNN         0.44        *0.59
# ------------------------------------------
# 10 Days  RF         0.47         0.63
#         SVC         0.46         0.61
#         NB          0.51         0.63
#         LSTM       *0.62        *0.68
#         CNN         0.41         0.63
