############## Deeplearning ##########
#install.packages('quantmod') # https://www.nbshare.io/notebook/233979015/How-To-Analyze-Yahoo-Finance-Data-With-R/
library('quantmod') # 주가데이터 불러오기
library('lubridate') # ymd 등 날짜 데이터 처리


# 다우존스의 주가데이터 불러오기
price_data <- getSymbols('^DJI',from = '2011-12-31', to = '2022-05-01',src='yahoo',auto.assign=FALSE)
price_data <- as.data.frame(price_data)

# 개장일만 추출
opening = rownames(price_data)
opening = as.data.frame(opening)
dimnames(opening)[[2]] = c("Date")
opening$Date = as.Date(opening$Date) # 시간 데이터로 변경
opening$opening_date = 1 # opening_date  변수 생성 값은 1(개장일이라는 뜻)

# 12년부터 22년까지의 날짜를 모두 추출
Date = seq(ymd('2011-12-30'), ymd('2022-04-30'), by ='1 days')
Date = as.data.frame(Date)

# 모든 날짜 데이터와 개장일 결합, 모든 날짜 중 개장일이 아닌 경우는 0
news_data_date = merge(x = Date, y = opening, by = "Date", all.x = TRUE)
news_data_date$opening_date[is.na(news_data_date$opening_date)]=0 # 즉, 개장일은 1, 그렇지 않은 날은 0으로 표시

# 오늘의 뉴스는 다음 날을 예측하는데 사용
# 주말이나 공휴일의 경우 그 다음 개장일에 주가를 예측하는데 사용
# price_date는 오늘의 뉴스가 예측할 주가의 날짜
standard = news_data_date$Date[1] + days(1)
news_data_date$price_date = news_data_date$Date[1]
start = Sys.time()
for(i in seq(dim(news_data_date)[1],1, by=(-1))){
  if(i==dim(news_data_date)[1]){
    standard = news_data_date$Date[i]
    news_data_date$price_date[i]=standard + days(1)
  } else if((news_data_date$opening_date[i]==1)&(news_data_date$opening_date[i+1]==1)){
    standard = news_data_date$Date[i]
    news_data_date$price_date[i] = standard + days(1)
  } else if(i!=1){
    if((news_data_date$opening_date[i]==1)&(news_data_date$opening_date[i+1]==0)&(news_data_date$opening_date[i-1]==0)){
      news_data_date$price_date[i]=standard + days(1)
      standard = news_data_date$Date[i]
    } else {news_data_date$price_date[i]=standard}
  } else{news_data_date$price_date[i]=standard}
}
end = Sys.time()
end-start   # 3sec

save_folder <- "/home/whfhrs3260/R_code/csv_data/"
score = readRDS(file=paste0(save_folder,"score.RData"))

# score과 개장일 데이터 결합
score = as.data.frame(score)
score$Date = as.Date(score$Date)
score = merge(x = news_data_date, y = score, by = "Date", all.x = TRUE)

score = subset(score, select=c("Date","Score","price_date"))
score$Score = as.integer(score$Score)

## python의 감성분석의 범위가 -1, 1이므로 -1, 1 범위로 scale -> But 굳이 필요한지 모르겠음
# scaler = function(vector){
#   abs_int = abs(vector)
#   int_max = max(abs_int,na.rm=TRUE)
#   int_min = min(abs_int,na.rm=TRUE)
#   result = c()
#   for(i in 1:length(vector)){
#   if(1==sign(vector[i])|is.na(vector[i])){
#     result[i] = (abs_int[i]-int_min)/(int_max-int_min)
#   }else{result[i] = -(abs_int[i]-int_min)/(int_max-int_min)}
#   }
#   return(result)
# }
# 
# score$Score = scaler(score$Score)

# 예측할 주가의 날짜 기준으로 score 평균 계산 -> 즉, 예측할 날짜에 쓰일 뉴스의 감성점수를 평균 냄
score = score %>% 
  group_by(price_date) %>%
  summarise("Score"=mean(Score, na.rm = TRUE))

score = as.data.frame(score[score$price_date<="2022-04-29",])

first_data = getSymbols('^DJI',from = '2011-12-30', to = '2012-01-01',src='yahoo',auto.assign=FALSE)

price_data$Date = rownames(price_data)
colnames(score) = c("Date","Score")

price_data = price_data[(price_data$Date<"2013-01-01")&("2012-01-01"<=price_data$Date),]

price_data$Date = as.Date(price_data$Date)
price_data = merge(x = price_data, y = score, by = "Date", all.x = TRUE)

price_data$Score[is.na(price_data$Score)]=0

price_data$Today_Close = 0
for(i in 1:(dim(price_data)[1]-1)){
  price_data$Today_Close[i+1] = price_data$DJI.Close[i]
}
price_data$Today_Close[1]=first_data$DJI.Close

price_data = price_data[c("DJI.Close","Score","Today_Close")]
colnames(price_data) = c("Next_Close", "Score","Today_Close")



x = price_data[c("Score","Today_Close")]
y = price_data[c("Next_Close")]

