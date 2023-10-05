# Stock Market Forecasting Using Machine Learning Algorithms, 2012 -> news score을 포함해 실험한 결과는 Python 폴더에 수록
# 의문점 1 : 리뷰 논문에서는 이 논문을 예측 정확도가 0.77이고 window size를 최적화하면 0.85까지 올라간다고 소개
#            실제로 논문을 따라 해본 결과 1 day 부터 t day 까지의 여러 변수들의 변동 결과를 가지고 t day의 dow jones의 변동을 예측 했을 때 0.7~0.8의 정확도를 보임
#            하지만 1 day 부터 t day 까지의 여러 변수들의 변동 결과를 가지고 t+1 day의 dow jones의 변동을 예측했을 때 0.5 정도의 결과가 나옴  ==> ????
# 의문점 2 : 논문에서는 연관성이 높은 변수들만 사용했을 때 0.74의 결과를 얻었고, 모든 변수를 사용했을 때 0.63의 결과를 얻었다.
#            하지만 나는 연관성이 높은 변수들로 추렸을 때 정확도가 떨어진다.  ==> ???

# install.pakages('e1071')
library('quantmod')
library('tidyverse')
library('e1071')

# 수집할 날짜
start = "2011-12-30"
end = "2022-05-03"

# 총 16개의 주가 수집
# dow jones, NASDAQ, S&P500, Nekkei 255, Hang Seng Index, FTSE100, DAX, ASX, EUR, AUD, JPY, USD, Silver, Platinum, Oil, Gold
dow_jones <- getSymbols('^DJI',from = start, to = end,src='yahoo',auto.assign=FALSE)
dow_jones <- as.data.frame(dow_jones$DJI.Close)
dow_jones$Date <- rownames(dow_jones)
dow_jones = na.omit(dow_jones)

nasdaq <- getSymbols('^IXIC',from = start, to = end,src='yahoo',auto.assign=FALSE)
nasdaq <- as.data.frame(nasdaq$IXIC.Close)
nasdaq$Date <- rownames(nasdaq)
nasdaq = na.omit(nasdaq)

sp500 <- getSymbols('^GSPC',from = start, to = end ,src='yahoo',auto.assign=FALSE)
sp500 <- as.data.frame(sp500$GSPC.Close)
sp500$Date <- rownames(sp500)
sp500 = na.omit(sp500)

nekkei <- getSymbols('^N225',from = start, to = end,src='yahoo',auto.assign=FALSE)
nekkei <- as.data.frame(nekkei$N225.Close)
nekkei$Date <- rownames(nekkei)
nekkei = na.omit(nekkei)

hang_seng <- getSymbols('^HSI',from = start, to = end,src='yahoo',auto.assign=FALSE)
hang_seng <- as.data.frame(hang_seng$HSI.Close)
hang_seng$Date <- rownames(hang_seng)
hang_seng = na.omit(hang_seng)

FTSE <- getSymbols('^FTSE',from = start, to = end,src='yahoo',auto.assign=FALSE)
FTSE <- as.data.frame(FTSE$FTSE.Close)
FTSE$Date <- rownames(FTSE)
FTSE = na.omit(FTSE)

dax <- getSymbols('^GDAXI',from = start, to = end,src='yahoo',auto.assign=FALSE)
dax <- as.data.frame(dax$GDAXI.Close)
dax$Date <- rownames(dax)
dax = na.omit(dax)

asx <- getSymbols('^AXJO',from = start, to =end,src='yahoo',auto.assign=FALSE)
asx <- as.data.frame(asx$AXJO.Close)
asx$Date <- rownames(asx)
asx = na.omit(asx)


eur <- getSymbols('EURUSD=X',from = start, to = end,src='yahoo',auto.assign=FALSE)
eur <- as.data.frame(eur$`EURUSD=X.Close`)
eur$Date <- rownames(eur)
eur = na.omit(eur)

aud <- getSymbols('AUDUSD=X',from = start, to = end,src='yahoo',auto.assign=FALSE)
aud <- as.data.frame(aud$`AUDUSD=X.Close`)
aud$Date <- rownames(aud)
aud = na.omit(aud)

jpy <- getSymbols('JPYUSD=X',from = start, to = end,src='yahoo',auto.assign=FALSE)
jpy <- as.data.frame(jpy$`JPYUSD=X.Close`)
jpy$Date <- rownames(jpy)
jpy = na.omit(jpy)

usd <- getSymbols('USD',from = start, to = end,src='yahoo',auto.assign=FALSE)
usd <- as.data.frame(usd$USD.Close)
usd$Date <- rownames(usd)
usd = na.omit(usd)


gold <- getSymbols('GC=F',from = start, to = end,src='yahoo',auto.assign=FALSE)
gold <- as.data.frame(gold$`GC=F.Close`)
gold$Date <- rownames(gold)
gold = na.omit(gold)

silver <- getSymbols('SI=F',from = start, to = end,src='yahoo',auto.assign=FALSE)
silver <- as.data.frame(silver$`SI=F.Close`)
silver$Date <- rownames(silver)
silver = na.omit(silver)

platinum <- getSymbols('PL=F',from = start, to = end,src='yahoo',auto.assign=FALSE)
platinum <- as.data.frame(platinum$`PL=F.Close`)
platinum$Date <- rownames(platinum)
platinum = na.omit(platinum)


oil <- getSymbols('CL=F',from = start, to = end,src='yahoo',auto.assign=FALSE)
oil <- as.data.frame(oil$`CL=F.Close`)
oil$Date <- rownames(oil)
oil = na.omit(oil)

# Dow Jones의 Date 기준으로 16개의 변수 모두 합침(다른 변수들의 날짜를 무시하고 Dow Jones 기준으로 맞춤)
input_feature <- list(dow_jones, nasdaq, sp500, nekkei, hang_seng, FTSE, dax, asx, eur, aud, jpy, usd, gold, silver, platinum, oil) %>% reduce(left_join, by = "Date")
# Date를 제거하고, 선형보간법으로 비어있는 주가를 채움
input_feature <- apply(subset(input_feature, select = -Date), 2, function(x){na.approx(x)[1:dim(dow_jones)[1]]})

# Data Frame으로 변경
input_feature <- as.data.frame(input_feature)
# 열 이름 재생성
colnames(input_feature) <- c("DOWJ", "NASDAQ", "SP500", "Nikkei", "Hangseng", "FTSE", "DAX", "ASX", "EUR", "AUD", "JPY", "USD", "Gold", "Silver", "Platinum", "Oil")

# input feature : input date 입력
# difference : t day 주가 - t-d day 주가의 오르내림을 예측하기 위해 d를 difference로 입력
# case : 0은 t day 주가 - t-d day 주가 예측, 1은 t+d day 주가 - t day 주가 예측, 나머지는 error
# 정규화를 하여 input data 구성 후 train을 70%, test 30% 구분
# 1. 변수 각각 input data로 사용하여 SVM 실행 -> dowjones와 연관성 알아봄
# 2. Dow Jones를 제외한 15개의 변수를 이용해 두 가지 경우로 나누어 SVM 적용 1) 모든 변수 2) 연관성이 높은 4개의 변수 
# 3. DOW Jones, NASDAQ, S&P 500을 제외한 총 13개의 변수를 이용해 Dow Jones를 예측하기 위한 SVM 적용, 2번과 마찬가지로 두 가지 경우로 나눔

svm_accuracy = function(input_feature, difference, case){
  # 입력한 difference 에 따라 정규화 진행
  input_normal <- (input_feature - lag(input_feature, difference))/lag(input_feature, difference)
  input_normal <- input_normal/abs(input_normal)
  input_normal <- input_normal[(difference+1):dim(input_normal)[1],]
  
  # t day 주가와 t-d day 주가가 동일한 경우가 있음 -> 0으로 처리
  input_normal[is.na(input_normal)] <- 0
  
  if(case == 0){
    # 입력변수 : t day 주가 - t-d day 주가 => 출력 변수 : t day 주가 - t-d day 주가
  } else if(case == 1){
    # 입력변수 : t day 주가 - t-d day 주가 => 출력 변수 : t+d day 주가 - t day 주가
    input_normal[,2:dim(input_normal)[2]] <- apply(input_normal[,2:dim(input_normal)[2]],2, lag)
    input_normal <- input_normal[2:dim(input_normal)[1],]
  } else {
     stop()
    }
  
  # train, test data 구성
  train_ind <- dim(input_normal)[1]*0.7
  train <- input_normal[seq(1, train_ind),]
  test <- input_normal[seq(train_ind+1, dim(input_normal)[1]),]
  
  # 1. 변수 각각 input data로 사용하여 SVM 실행
  Nasdaq_svm <- svm(train$NASDAQ, train$DOWJ, type = "C-classification")
  SP500_svm <- svm(train$SP500, train$DOWJ, type = "C-classification")
  Nikkei_svm <- svm(train$Nikkei, train$DOWJ, type = "C-classification")
  Hangseng_svm <- svm(train$Hangseng, train$DOWJ, type = "C-classification")
  FTSE_svm <- svm(train$FTSE, train$DOWJ, type = "C-classification")
  DAX_svm <- svm(train$DAX, train$DOWJ, type = "C-classification")
  ASX_svm <- svm(train$ASX, train$DOWJ, type = "C-classification")
  EUR_svm <- svm(train$EUR, train$DOWJ, type = "C-classification")
  AUD_svm <- svm(train$AUD, train$DOWJ, type = "C-classification")
  JPY_svm <- svm(train$JPY, train$DOWJ, type = "C-classification")
  USD_svm <- svm(train$USD, train$DOWJ, type = "C-classification")
  Gold_svm <- svm(train$Gold, train$DOWJ, type = "C-classification")
  Silver_svm <- svm(train$Silver, train$DOWJ, type = "C-classification")
  Platinum_svm <- svm(train$Platinum, train$DOWJ, type = "C-classification")
  Oil_svm <- svm(train$Oil, train$DOWJ, type = "C-classification")
  
  # 만든 SVM으로 test data 예측
  Nasdaq_pre <- predict(Nasdaq_svm, test$NASDAQ)
  SP500_pre <- predict(SP500_svm, test$SP500)
  Nikkei_pre <- predict(Nikkei_svm, test$Nikkei)
  Hangseng_pre <- predict(Hangseng_svm, test$Hangseng)
  FTSE_pre <- predict(FTSE_svm, test$FTSE)
  DAX_pre <- predict(DAX_svm, test$DAX)
  ASX_pre <- predict(ASX_svm, test$ASX)
  EUR_pre <- predict(EUR_svm, test$EUR)
  AUD_pre <- predict(AUD_svm, test$AUD)
  JPY_pre <- predict(JPY_svm, test$JPY)
  USD_pre <- predict(USD_svm, test$USD)
  Gold_pre <- predict(Gold_svm, test$Gold)
  Silver_pre <- predict(Silver_svm, test$Silver)
  Platinum_pre <- predict(Platinum_svm, test$Platinum)
  Oil_pre <- predict(Oil_svm, test$Oil)
  
  # 각 accuracy 저장
  predict_dowjones <- data.frame("NASDAQ"=c(0))
  predict_dowjones$NASDAQ <- sum(as.vector(Nasdaq_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$SP500 <- sum(as.vector(SP500_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$Nikkei <- sum(as.vector(Nikkei_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$Hangseng <- sum(as.vector(Hangseng_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$FTSE <- sum(as.vector(FTSE_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$DAX <- sum(as.vector(DAX_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$ASX <- sum(as.vector(ASX_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$EUR <- sum(as.vector(EUR_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$AUD <- sum(as.vector(AUD_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$JPY <- sum(as.vector(JPY_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$USD <- sum(as.vector(USD_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$Gold <- sum(as.vector(Gold_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$Silver <- sum(as.vector(Silver_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$Platinum <- sum(as.vector(Platinum_pre) == test$DOWJ)/length(test$DOWJ)
  predict_dowjones$Oil <- sum(as.vector(Oil_pre) == test$DOWJ)/length(test$DOWJ)
  
  # 15개의 변수를 사용하고 모든 변수를 그대로 사용한 경우 
  train_all_15 <- subset(train, select = -c(DOWJ))
  train_y <- train$DOWJ
  test_all_15 <- subset(test, select = -c(DOWJ))
  test_y <- test$DOWJ
  
  svm_all_15 <- svm(train_all_15, train_y,  type = "C-classification")
  pre_all_15 <- predict(svm_all_15, test_all_15)
  accuracy_all_15 <- sum(as.vector(pre_all_15) == test_y)/length(test_y)
  
  # 15개의 변수를 사용하고 연관성이 높은 4개의 변수만 추려서 사용한 경우
  train_four_15 <- subset(train, select = colnames(sort(predict_dowjones,decreasing = TRUE)[1:4]))
  test_four_15 <- subset(test, select = colnames(sort(predict_dowjones,decreasing = TRUE)[1:4]))
  
  svm_four_15 <- svm(train_four_15, train_y, type = "C-classification")
  pre_four_15 <- predict(svm_four_15, test_four_15)
  accuracy_four_15 <- sum(as.vector(pre_four_15) == test_y)/length(test_y)
  
  # NASDAQ과 S&P500 제거
  predict_dowjones <- subset(predict_dowjones, select = -c(NASDAQ, SP500))
  
  # 13개의 변수를 사용하고 모든 변수를 그대로 사용한 경우
  train_all_13 <- subset(train, select = -c(DOWJ, NASDAQ, SP500))
  test_all_13 <- subset(test, select = -c(DOWJ, NASDAQ, SP500))
  
  svm_all_13 <- svm(train_all_13, train_y,  type = "C-classification")
  pre_all_13 <- predict(svm_all_13, test_all_13)
  accuracy_all_13 <- sum(as.vector(pre_all_13) == test_y)/length(test_y)
  
  # 13개의 변수를 사용하고 연고나성이 높은 4개의 변수만 추려서 사용한 경우
  train_four_13 <- subset(train, select = colnames(sort(predict_dowjones,decreasing = TRUE)[1:4]))
  test_four_13 <- subset(test, select = colnames(sort(predict_dowjones,decreasing = TRUE)[1:4]))
  
  svm_four_13 <- svm(train_four_13, train_y, type = "C-classification")
  pre_four_13 <- predict(svm_four_13, test_four_13)
  accuracy_four_13 <- sum(as.vector(pre_four_13) == test_y)/length(test_y)

  return(c(accuracy_all_15, accuracy_four_15, accuracy_all_13, accuracy_four_13))
}

# case = 0 즉, 입력 변수 : t day 주가 - t-d day 주가, 출력 변수 : t day 주가 - t-d day Dow Jones 주가
difference <- 1:40
accuracy_result_0 <- sapply(difference, function(x){svm_accuracy(input_feature, x, 0)})
par(mfrow=c(2,2))
plot(accuracy_result_0[1,], type="l", main = "15 feature all", ylab = "accuracy", xlab = "difference", col = "red", ylim=c(0.7, 0.95))
plot(accuracy_result_0[2,], type="l", main = "15 feature four", ylab = "accuracy", xlab = "difference", col = "blue", ylim=c(0.7, 0.95))
plot(accuracy_result_0[3,], type="l", main = "13 feature all", ylab = "accuracy", xlab = "difference", col = "orange", ylim=c(0.7, 0.95))
plot(accuracy_result_0[4,], type="l", main = "13 feature four", ylab = "accuracy", xlab = "difference", col = "black", ylim=c(0.7, 0.95))

par(mfrow = c(1,1))
plot(accuracy_result_0[1,], type="l",main = "t day prediction", ylab = "accuracy", xlab = "difference", col = "red", ylim=c(0.7, 0.95), lwd=2)
points(accuracy_result_0[2,], type="l", ylab = "accuracy", xlab = "difference", col = "blue", ylim=c(0.7, 0.95), lwd=2,lty=3)
points(accuracy_result_0[3,], type="l", ylab = "accuracy", xlab = "difference", col = "orange", ylim=c(0.7, 0.95), lwd=2)
points(accuracy_result_0[4,], type="l", ylab = "accuracy", xlab = "difference", col = "black", ylim=c(0.7, 0.95), lwd=2,lty=3)
legend("topleft", legend = c("15 feature all", "15 feature four", "13 feature all", "13 feature four"), fill = c("red", "blue","orange", "black"),border="white",box.lty=0,cex=0.75)

# case = 1 즉, 입력 변수 : t day 주가 - t-d day 주가, 출력 변수 : t+d day 주가 - t day Dow Jones 주가
difference <- 1:100
accuracy_result_1 <- sapply(difference, function(x){svm_accuracy(input_feature, x, 1)})

par(mfrow=c(2,2))
plot(accuracy_result_1[1,], type="l", main = "15 feature all", ylab = "accuracy", xlab = "difference", col = "red", ylim=c(0.5, 0.95))
plot(accuracy_result_1[2,], type="l", main = "15 feature four", ylab = "accuracy", xlab = "difference", col = "blue", ylim=c(0.5, 0.95))
plot(accuracy_result_1[3,], type="l", main = "13 feature all", ylab = "accuracy", xlab = "difference", col = "orange", ylim=c(0.5, 0.95))
plot(accuracy_result_1[4,], type="l", main = "13 feature four", ylab = "accuracy", xlab = "difference", col = "black", ylim=c(0.5, 0.95))

par(mfrow = c(1,1))
plot(accuracy_result_1[1,], type="l",main = "t+d day prediction", ylab = "accuracy", xlab = "difference", col = "red", ylim=c(0.5, 0.95), lwd=2)
points(accuracy_result_1[2,], type="l", ylab = "accuracy", xlab = "difference", col = "blue", ylim=c(0.5, 0.95), lwd=2,lty=3)
points(accuracy_result_1[3,], type="l", ylab = "accuracy", xlab = "difference", col = "orange", ylim=c(0.5, 0.95), lwd=2)
points(accuracy_result_1[4,], type="l", ylab = "accuracy", xlab = "difference", col = "black", ylim=c(0.5, 0.95), lwd=2,lty=3)
legend("topleft", legend = c("15 feature all", "15 feature four", "13 feature all", "13 feature four"), fill = c("red", "blue","orange", "black"),border="white",box.lty=0,cex=0.75)



# difference를 무지막지하게 늘리면? 
svm_accuracy(input_feature, 700, 1)

input_normal <- (input_feature - lag(input_feature, 700))/lag(input_feature, 700)
input_normal <- input_normal/abs(input_normal)
input_normal <- input_normal[(700+1):dim(input_normal)[1],]

# t day 주가와 t-d day 주가가 동일한 경우가 있음 -> 0으로 처리
input_normal[is.na(input_normal)] <- 0

if(case == 0){
  # 입력변수 : t day 주가 - t-d day 주가 => 출력 변수 : t day 주가 - t-d day 주가
} else if(case == 1){
  # 입력변수 : t day 주가 - t-d day 주가 => 출력 변수 : t+d day 주가 - t day 주가
  input_normal[,2:dim(input_normal)[2]] <- apply(input_normal[,2:dim(input_normal)[2]],2, lag)
  input_normal <- input_normal[2:dim(input_normal)[1],]
} else {
  stop()
}

# train, test data 구성
train_ind <- dim(input_normal)[1]*0.7
train <- input_normal[seq(1, train_ind),]
test <- input_normal[seq(train_ind+1, dim(input_normal)[1]),]

