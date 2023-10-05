# install.packages("randomForest")
library('quantmod') # 주가데이터 불러오기
library('lubridate') # ymd 등 날짜 데이터 처리
library(randomForest)


score = read_csv(file="R_code/nyt_data/score.csv")

price_data <- getSymbols('^DJI',from = '2011-12-30', to = '2022-05-01',src='yahoo',auto.assign=FALSE)
price_data <- as.data.frame(price_data)

opening = rownames(price_data)
opening = as.data.frame(opening)
dimnames(opening)[[2]] = c("Date")
opening$Date = as.Date(opening$Date) # 시간 데이터로 변경
opening$opening_date = 1 # opening_date  변수 생성 값은 1(개장일이라는 뜻)

# 12년부터 22년까지의 날짜를 모두 추출
Date = seq(ymd('2011-12-30'), ymd('2022-04-30'), by ='1 days')
Date = as.data.frame(Date)

news_data_date = merge(x = Date, y = opening, by = "Date", all.x = TRUE)
news_data_date$opening_date[is.na(news_data_date$opening_date)]=0 # 즉, 개장일은 1, 그렇지 않은 날은 0으로 표시

standard = news_data_date$Date[1] + days(1)
news_data_date$price_date = news_data_date$Date[1]

for(i in seq(dim(news_data_date)[1],1, by=(-1))){
  if(i==dim(news_data_date)[1]){
    news_data_date$price_date[i]=standard
    standard = news_data_date$Date[i]
  } else if(news_data_date$opening_date[i]==1){
    news_data_date$price_date[i] = standard
    standard = news_data_date$Date[i]
  } else{news_data_date$price_date[i]=standard}
}

# score과 개장일 데이터 결합
score = as.data.frame(score)
score$Date = as.Date(score$Date)
score = merge(x = news_data_date, y = score, by = "Date", all.x = TRUE)
score = subset(score, select=c("Date","Score","price_date"))


score = score %>% 
  group_by(price_date) %>%
  summarise("Score"=mean(Score, na.rm = TRUE))

score = as.data.frame(score[score$price_date<="2022-04-30",])

first_data = getSymbols('^DJI',from = '2011-12-29', to = '2011-12-30',src='yahoo',auto.assign=FALSE)

price_data$Date = rownames(price_data)
colnames(score) = c("Predict_Date","Score")

price_data$Date = as.Date(price_data$Date)
price_data = merge(x = price_data, y = score, by.x = "Date", by.y = "Predict_Date", all.x = TRUE)

price_data$Score[is.na(price_data$Score)]=0

price_data$Today_Close = 0
for(i in 1:(dim(price_data)[1]-1)){
  price_data$Today_Close[i+1] = price_data$DJI.Close[i]
}
price_data$Today_Close[1]=first_data$DJI.Close

price_data = price_data[c("DJI.Close","Score","Today_Close")]
colnames(price_data) = c("Predict_Close", "Score","Today_Close")

price_data$Predict_diff = 0
price_data$Today_diff = 0
price_data$Predict_diff = price_data$Predict_Close - lag(price_data$Predict_Close)
price_data$Today_diff = price_data$Today_Close - lag(price_data$Today_Close)

price_data$Predict_diff = lapply(price_data$Predict_diff, function(x){
  if(is.na(x)){
  } else if(x>=0){
    x = 1
  } else {x=0}
  })

price_data$Today_diff = lapply(price_data$Today_diff, function(x){
  if(is.na(x)){
  } else if(x>=0){
    x = 1
  } else {x=0}
})

price_data = price_data[2:dim(price_data)[1],]
min = min(price_data$Today_Close)
max = max(price_data$Today_Close)
price_data$Today_Close = lapply(price_data$Today_Close, function(x){ (x-min)/(max-min)})
price_data <- price_data[c("Score","Today_Close","Today_diff","Predict_diff")]
price_data = apply(price_data, 2, function(x){unlist(x)})

train_ind = as.integer(dim(price_data)[1]*0.7)

train = price_data[1:train_ind,]
test = price_data[train_ind:dim(price_data)[1],]

RF <- randomForest(Predict_diff ~., data=train,ntree=50)
pred = predict(RF, newdata = test, type="class")
pred = lapply(pred, function(x){
  if(x>0.5){
    x=1
  }else{x=0}
})
sum(pred==test[,4])/nrow(test)

