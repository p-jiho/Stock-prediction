################### preprocessing ############
## package
# install.packages("tidytext")   # 토큰화, 감성분석
# install.packages("tidyverse")
# install.packages("textstem")   # 표제어추출
# install.packages("stopwords")  # 불용어
# install.packages("parallel")
# install.packages("syuzhet")
require("tidytext")
require("tidyverse")
require("textstem")
require("stopwords")
require("parallel")
library('syuzhet') #감정사전


# news data 파일 이름 불러오기
dir = "./data"
year_month = list.files(dir)


preprocessing = function(yearmonth){
  
  # data read
  dir = "./data"
  news_data = readLines(paste(dir,"/", year_month[1], sep=""))
  
  # 5가지 항목으로 분리
  news_data = strsplit(news_data, split = '---')
  
  lct <- Sys.getlocale("LC_TIME"); Sys.setlocale("LC_TIME", "C")  # as.Date의 결과가 출력되도록 만듦
  for(i in 1:length(news_data)){
    # 날짜를 최초작성일로 구성
    news_data[[i]][2] = strsplit(news_data[[i]][2], split="UTC")[[1]][1]
    
    # title과 text data 결합
    news_data[[i]][3] = paste(news_data[[i]][3], news_data[[i]][4], sep=" ")
    
    # 날짜가 없거나 날짜가 2012년 미만 2022년 5월 이후인 경우 데이터 NA로 변환
    # "yyyy-mm-dd" 형식으로 변환
    if(identical(intersect(substr(yearmonth, 6, 7),c("01","02", "08","09","10","11","12")),character(0))){
      news_data[[i]][2] = as.character(as.Date(news_data[[i]][2], format=" %b %d, %Y, %I:%M %p "))
    }else{
      news_data[[i]][2] = gsub("Sept", "Sep", news_data[[i]][2]) # Sept를 규정에 맞게 Sep로 변환
      news_data[[i]][2] = as.character(as.Date(news_data[[i]][2], format=" %B. %d, %Y, %I:%M %p "))
    }
    
    # 불필요한 문장부호 제거
    news_data[[i]][3] = gsub(pattern = "[,—\"“”‘’\`\':!;?\\-]", replacement = "", x=news_data[[i]][3])
    
    # 값이 없는 데이터는 삭제
    if(is.na(news_data[[i]][2])){
      news_data[[i]]=NA
    }else if(news_data[[i]][2]<"2012-01-01"|news_data[[i]][2]>="2022-05-01"){
      news_data[[i]]=NA
    }else(news_data[[i]]=news_data[[i]][1:3])
    
  }
  
  # 날짜가 없거나 기준에 맞지 않아 NA로 변경된 데이터 삭제
  k=1
  for(i in 1:length(news_data)){
    if(length(is.na(news_data[[k]]))==1 ){
      if(is.na(news_data[[k]])){
        news_data[[k]]=NULL
      }else{k=k+1}
    }else{k=k+1}
  }
  
  # title과 text 결합된 데이터를 토큰화(단어로, 대문자를 소문자로 변환하지 않음, 문장부호 제거X) 
  text_token = list()
  for(i in 1:length(news_data)){
    text_token[i] = as.vector(unnest_tokens(tibble(text=news_data[[i]][3]),output = word, input = text, token = "words", to_lower = FALSE, strip_punct = FALSE))
  }
  
  # 소문자 변환
  text_token = lapply(text_token, upper_to_lower_line)
  
  # 표제어 추출
  text_token = lapply(text_token, lemmatize)
  
  
  # 불용어 제거
  stopword = stopwords::stopwords("en", source = "nltk")
  text_token = lapply(text_token, diff, stopword)
  
  # result = list(news_data=news_data, text_token = text_token)
  return(text_token)
}

# 소문자 변환
upper_to_lower_word = function(word){
  word = gsub(pattern = "^[A-Z]{1}[a-z]*$", replacement = tolower(word), x=word)
  return(word)
}

upper_to_lower_line = function(line){
  line = sapply(line, upper_to_lower_word)
  return(line)
}

# 표제어 추출
lemmatize = function(vector){
  return(as.vector(lemmatize_words(vector)))
}

# 불용어 제거
diff = function(line, stopwords){
  stopword = intersect(stopwords, line)
  token=c()
  for(i in 1:length(line)){
    if(!(line[i] %in% stopword)){
      token = c(token,line[i])
    }
  }
  return(token)
}

# parallel
#https://cinema4dr12.tistory.com/1023
#https://hoontaeklee.github.io/en/posts/20200607_r%EB%B3%91%EB%A0%AC%EC%B2%98%EB%A6%AC%EB%A9%94%EB%AA%A8/

# core 수
numcore = detectCores()-1
clus = makeCluster(numcore, type = "PSOCK")

# 필요한 함수와 패키지 지정 => 안하면 인식하지 못함
clusterExport(clus,c("preprocessing", "upper_to_lower_line", "upper_to_lower_word", "lemmatize", "diff"))
clusterEvalQ(clus,require("tidytext"))
clusterEvalQ(clus,require("tidyverse"))
clusterEvalQ(clus,require("textstem"))
clusterEvalQ(clus,require("stopwords"))

start = Sys.time()
news_data_preprocessing = parLapply(cl=clus, X = year_month, fun = preprocessing)
end = Sys.time()
end-start # 1hour, 9min

stopCluster(clus)

save_folder <- "/home/whfhrs3260/R_code/csv_data/"

saveRDS(news_data_preprocessing,file=paste(save_folder,"news_data_preprocessing.RData"))
# news_data_preprocessing = readRDS(paste(save_folder,"news_data_preprocessing.RData"))

# news_data, text_token 결과 정리하기 (paraller 과정에서 규칙을 가지고 섞여있는 데이터를 분리)
start = Sys.time()
news_data = list()
text_token = list()
for(i in 1:length(news_data_preprocessing)){
  print(paste(i,":",length(news_data_preprocessing[[i]][[1]])))
  for(j in 1:length(news_data_preprocessing[[i]][[1]])){
    news_data = append(news_data, news_data_preprocessing[[i]][[1]][j])  # news_data
    text_token = append(text_token, news_data_preprocessing[[i]][[2]][j])  # tex_token
  }
}
end = Sys.time()
end-start # 1hour

#saveRDS(news_data,file=paste(save_folder,"news_data.RData"))
#saveRDS(text_token,file=paste(save_folder,"text_token.RData"))

#news_data = readRDS(file=paste(save_folder,"news_data.RData"))
#text_token = readRDS(file=paste(save_folder,"text_token.RData"))

# text_token의 형식을 list에서 vector로 변경
total_token = unlist(text_token)

# token 빈도수 추출
start = Sys.time()
total_token_count = table(total_token)
end = Sys.time()
end-start # 1min

# 빈도수 기준 오름차순 정렬
total_token_count = sort(total_token_count,decreasing = TRUE)

# 빈도수가 5이하인 단어 추출
total_token_less_frequent_word = names(total_token_count[5>=total_token_count])

# 변수 삭제
rm(total_token_count)

# 빈도수가 5이하인 단어 삭제
del_less_frequent_word = function(word_vector){
  return(word_vector[! word_vector %in% total_token_less_frequent_word])
}


# numcore = detectCores()-1
# clus = makeCluster(numcore, type = "PSOCK")

#clusterExport(clus,c("diff","total_token_less_frequent_word"))

#start = Sys.time()
#result = parLapply(cl=clus, X = text_token, fun = del_less_frequent_word)
#end = Sys.time()
#end-start # 약 12시간 예상(시간이 너무 오래걸려 돌려보지는 못함)

#stopCluster(clus)

# 토큰화 되어있는 단어들을 다시 문장으로 결합
start = Sys.time()
sentence = vector(mode="list",length=length(text_token))
for(i in 1:length(text_token)){
  sentence[[i]] = paste(text_token[[i]],collapse =" ")
}
end = Sys.time()
end-start # 52sec

# 시간 데이터 결합 즉, 데이터는 시간, news 문장 데이터로 구성
start = Sys.time()
sentence_time=list()
for(i in 1:length(news_data)){
  sentence_time[[i]] = c(news_data[[i]][2],sentence[[i]])
}
end = Sys.time()
end-start # 1sec

# 시간순으로 정렬
sentence_time = sentence_time[order(sapply(sentence_time, function(x) x[1], simplify=TRUE))]

# 감성분석(bing은 단어사전의 일종)
start = Sys.time()
for(i in 1:length(sentence_time)){
  sentence_time[[i]][3] = get_sentiment(sentence_time[[i]][2], method="bing")
}
end = Sys.time()
end-start   # 12min

# 데이터 프레임으로 변경(뒤 코드들은 데이터프레임으로 처리하기가 간편, NA 데이터도 없음)
start = Sys.time()
score = as.data.frame(sentence_time)
score = t(score)
end = Sys.time()
end-start   # 13min

# 행, 열 이름 새로 지정
dimnames(score)[[1]] = as.character(1:length(score)/dimnames(score)[[2]])
dimnames(score)[[2]] = c("Date","Title_Text", "Score")

# 시간과 감성점수만 추출
score = subset(score, select=c("Date","Score"))
write.csv(score,file=paste(save_folder,"/score.csv", sep=""))


