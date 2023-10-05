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
dir <- "./data"
year_month <- list.files(dir)

preprocessing = function(year_month){
  
  # data read
  dir <- "./data"
  news_data <- readLines(paste(dir,"/", year_month, sep=""))
  
  # 데이터를 R code로 생성한 경우
  # news_data <- str_sub(news_data,2,nchar(news_data)-1)
  
  # 5가지 항목으로 분리
  news_data <- strsplit(news_data, split = '---')
  
  lct <- Sys.getlocale("LC_TIME"); Sys.setlocale("LC_TIME", "C")  # as.Date의 결과가 출력되도록 만듦
  
  news_data <- lapply(news_data, function(x){
    x[2] <- strsplit(x[2], split="UTC")[[1]][1]
    x[3] <- paste(x[3], x[4], sep=" ")
    
    if(identical(intersect(substr(year_month, 6, 7),c("01","02", "08","09","10","11","12")),character(0))){
      x[2] <- as.character(as.Date(x[2], format=" %b %d, %Y, %I:%M %p "))
    }else{
      x[2] <- gsub("Sept", "Sep", x[2]) # Sept를 규정에 맞게 Sep로 변환
      x[2] <- as.character(as.Date(x[2], format=" %B. %d, %Y, %I:%M %p "))
    }
    
    x[3] <- gsub(pattern = "[,—\"“”‘’\`\':!;?\\-]", replacement = " ", x=x[3])
    
    # 값이 없는 데이터는 삭제
    if(is.na(x[2])){
      x <- NA
    }else if(x[2]<"2012-01-01"|x[2]>="2022-05-01"){
      x <- NA
    }else(x <- x[1:3])
  })
  
  # 날짜가 없거나 기준에 맞지 않아 NA로 변경된 데이터 삭제
  # discard 함수에서 is.na는 1가지의 값을 계산하여 삭제하므로 3가지의 값이 나타나서 사용을 못함
  news_data <- lapply(news_data, function(x){discard(x, is.na)})  # NA는 logical(0)으로 변환
  news_data <- discard(news_data, is.logical)           # logical 인 경우 삭제
  
  # title과 text 결합된 데이터를 토큰화(단어로, 대문자를 소문자로 변환하지 않음, 문장부호 제거X) 
  text_token <- lapply(news_data,function(x){as.vector(unnest_tokens(tibble(text=x[3]),output = word, input = text, token = "words", to_lower = FALSE, strip_punct = FALSE))})
  
  # 소문자 변환
  text_token <- lapply(text_token, function(line){
    sapply(line[[1]], function(word){gsub(pattern = "^[A-Z]{1}[a-z]*$", replacement = tolower(word), x=word)})
  })
  
  # 표제어 추출
  text_token <- lapply(text_token, function(vector){as.vector(lemmatize_words(vector))})
  
  
  # 불용어 제거
  stopword <- stopwords::stopwords("en", source = "nltk")
  text_token <- lapply(text_token, function(line){
    stopword <- intersect(stopword, line)
    sapply(line, function(word){
      if(!(word %in% stopword)){
        word
      }else{NA}
    })
  })
  
  text_token <- lapply(text_token, function(x){as.vector(discard(x,is.na))})
  
  result <- list(news_data=news_data, text_token = text_token)
  return(result)
}

# parallel
#https://cinema4dr12.tistory.com/1023
#https://hoontaeklee.github.io/en/posts/20200607_r%EB%B3%91%EB%A0%AC%EC%B2%98%EB%A6%AC%EB%A9%94%EB%AA%A8/

# core 수
numcore <- detectCores()-1
clus <- makeCluster(numcore, type = "PSOCK")

clusterEvalQ(clus,require("tidytext"))
clusterEvalQ(clus,require("tidyverse"))
clusterEvalQ(clus,require("textstem"))
clusterEvalQ(clus,require("stopwords"))

start <- Sys.time()
news_data_preprocessing <- parLapply(cl=clus, X = year_month, fun = preprocessing)
end <- Sys.time()
end-start # 1hour

stopCluster(clus)

getwd()
save_folder <- "./R_code/csv_data"
saveRDS(news_data_preprocessing,file=paste(save_folder,"/news_data_preprocessing.RData", sep=""))

# news_data, text_token 결과 정리하기 (paraller 과정에서 규칙을 가지고 섞여있는 데이터를 분리)
news_data <- unlist(lapply(news_data_preprocessing, function(list){list[[1]]}), recursive=FALSE)
text_token <- unlist(lapply(news_data_preprocessing, function(list){list[[2]]}), recursive=FALSE)


# text_token의 형식을 list에서 vector로 변경
total_token <- unlist(text_token)

# token 빈도수 추출
total_token_count <- table(total_token)

# 빈도수 기준 오름차순 정렬
total_token_count <- sort(total_token_count,decreasing = TRUE)

# 빈도수가 5이하인 단어 추출
total_token_less_frequent_word <- names(total_token_count[5>=total_token_count])

# 빈도수가 5이하인 단어 삭제
del_less_frequent_word <- function(word_vector){
  return(word_vector[! word_vector %in% total_token_less_frequent_word])
}

numcore <- detectCores()-1
clus <- makeCluster(numcore, type = "PSOCK")

clusterExport(clus,"total_token_less_frequent_word")

start <- Sys.time()
result <- parLapply(cl=clus, X = text_token[1:1000], fun = del_less_frequent_word)
end <- Sys.time()
end-start # 약 3시간 예상(시간이 너무 오래걸려 돌려보지는 못함)

stopCluster(clus)

# 토큰화 되어있는 단어들을 다시 문장으로 결합
start <- Sys.time()
sentence <- lapply(text_token, function(x){paste(x, collapse=" ")})
end <- Sys.time()
end-start # 52sec

# 시간 데이터 결합 즉, 데이터는 시간, news 문장 데이터로 구성
sentence_time <- list()
for(i in 1:length(news_data)){
  sentence_time[[i]] <- c(news_data[[i]][2],sentence[[i]])
}

# 시간순으로 정렬
sentence_time <- sentence_time[order(sapply(sentence_time, function(x) x[1], simplify=TRUE))]


# 감성분석(bing은 단어사전의 일종)
start <- Sys.time()
sentence_time <- lapply(sentence_time, function(x){x[3] = get_sentiment(x[2], method="bing"); return(x)})
end <- Sys.time()
end-start   # 2min

# 데이터 프레임으로 변경(뒤 코드들은 데이터프레임으로 처리하기가 간편, NA 데이터도 없음)
start <- Sys.time()
score <- as.data.frame(sentence_time)
score <- t(score)
end <- Sys.time()
end-start   # 13min

dimnames(score)[[1]] <- as.character(1:length(score)/dimnames(score)[[2]])
dimnames(score)[[2]] <- c("Date","Title_Text", "Score")

# 시간과 감성점수만 추출
score <- subset(score, select=c("Date","Score"))
write.csv(score,file=paste(save_folder,"/score.csv", sep=""))
