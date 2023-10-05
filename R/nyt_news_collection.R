library(jsonlite)

nyt_news_collection <- function(year_month){
  key <- "G7AS6LA9s8ElEiMAMNYKoWgAFFq4ouAJ"

  year <- year_month[2]
  month <- year_month[1]

  nyt_base_url <- paste("https://api.nytimes.com/svc/archive/v1/", year, "/", month, ".json?api-key=", sep = "")
  nyt_news_url <- paste(nyt_base_url, key, sep="")
  nyt_news <- fromJSON(nyt_news_url)

  nyt_headline_date <- as.list(transpose(merge(nyt_news$response$docs$headline$main, nyt_news$response$docs$pub_date)))
  nyt_headline_date <- lapply(nyt_headline_date, function(time){
    time[2] <- as.Date(time[2])
    if(time[2]<"2012-01-01"|time[2]>="2022-05-01"){
      time <- NA
    } else{time <- time}
    return(time)
  })
  
  # discard 함수에서 is.na는 1가지의 값을 계산하여 삭제하므로 3가지의 값이 나타나서 사용을 못함
  nyt_headline_date <- lapply(nyt_headline_date, function(x){discard(x, is.na)})  # NA는 logical(0)으로 변환
  nyt_headline_date <- discard(nyt_headline_date, is.logical) 

  nyt_headline_date <- sapply(nyt_headline_date, function(line){gsub(pattern = "[,—\"“”‘’\`\':!;?\\-]", replacement = " ", x=line)})

  nyt_headline_token <- sapply(nyt_headline_date,function(x){as.vector(unnest_tokens(tibble(text=x[1]),output = word, input = text, token = "words", to_lower = FALSE, strip_punct = FALSE))})

  nyt_headline_token <- lapply(nyt_headline_token, function(line){
    sapply(line, function(word){gsub(pattern = "^[A-Z]{1}[a-z]*$", replacement = tolower(word), x=word)})
  })

  nyt_headline_token <- lapply(nyt_headline_token, function(vector){as.vector(lemmatize_words(vector))})

  stopword <- stopwords::stopwords("en", source = "nltk")
  nyt_headline_token <- lapply(nyt_headline_token, function(line){
    stopword <- intersect(stopword, line)
    sapply(line, function(word){
      if(!(word %in% stopword)){
        word
      }else{NA}
    })
  })
  nyt_headline_token <- lapply(nyt_headline_token, function(x){as.vector(discard(x,is.na))})
  
  result <- list(nyt_date=nyt_headline_date, nyt_headline_token = nyt_headline_token)
  return(result)
}

years = seq(2012, 2022)
months = seq(1, 12)
year_month = as.list(transpose(merge(months, years)[1:124,]))
start <- Sys.time()
test = nyt_news_collection(year_month[[1]])
end <- Sys.time()
end-start # 1hour

numcore <- 5
clus <- makeCluster(numcore, type = "PSOCK")

clusterEvalQ(clus,require("tidytext"))
clusterEvalQ(clus,require("tidyverse"))
clusterEvalQ(clus,require("textstem"))
clusterEvalQ(clus,require("stopwords"))
clusterEvalQ(clus,require("jsonlite"))

start <- Sys.time()
nyt_news_preprocessing <- parLapply(cl=clus, X = year_month[1:5], fun = nyt_news_collection)
end <- Sys.time()
end-start # 1hour

stopCluster(clus)

save_folder <- "./R_code/nyt_data"
saveRDS(nyt_news_preprocessing,file=paste(save_folder,"/nyt_news_preprocessing.RData", sep=""))

nyt_date <- unlist(lapply(nyt_news_preprocessing, function(list){list[[1]]}), recursive=FALSE)
nyt_headline_token <- unlist(lapply(nyt_news_preprocessing, function(list){list[[2]]}), recursive=FALSE)
names(nyt_date) <- c()
names(nyt_headline_token) <- c()

# text_token의 형식을 list에서 vector로 변경
nyt_total_token <- unlist(nyt_headline_token)

# token 빈도수 추출
nyt_total_token_count <- table(nyt_total_token)

# 빈도수 기준 오름차순 정렬
nyt_total_token_count <- sort(nyt_total_token_count,decreasing = TRUE)

# 빈도수가 5이하인 단어 추출
total_token_less_frequent_word <- names(nyt_total_token_count[5>=nyt_total_token_count])

# 빈도수가 5이하인 단어 삭제
del_less_frequent_word <- function(word_vector){
  return(word_vector[! word_vector %in% total_token_less_frequent_word])
}

numcore <- detectCores()-1
clus <- makeCluster(numcore, type = "PSOCK")

clusterExport(clus,"total_token_less_frequent_word")

start <- Sys.time()
nyt_headline_token <- parLapply(cl=clus, X = nyt_headline_token, fun = del_less_frequent_word)
end <- Sys.time()
end-start # 약 6 mins

stopCluster(clus)

# 토큰화 되어있는 단어들을 다시 문장으로 결합
start <- Sys.time()
nyt_sentence <- lapply(nyt_headline_token, function(x){paste(x, collapse=" ")})
end <- Sys.time()
end-start # 3sec

nyt_date <- sapply(nyt_date, function(time){
  time <- as.character(as.Date(time))
  if(time<"2012-01-01"|time>="2022-05-01"){
    time <- NA
  } else{time <- time}
})

nyt_sentence <- nyt_sentence[!is.na(nyt_date)]
nyt_date <- nyt_date[!is.na(nyt_date)]

# 시간 데이터 결합 즉, 데이터는 시간, news 문장 데이터로 구성
for(i in 1:length(nyt_sentence)){
  nyt_sentence[[i]] = c(nyt_date[i],nyt_sentence[[i]])
}

# 시간순으로 정렬
nyt_sentence <- nyt_sentence[order(sapply(nyt_sentence, function(x) x[1], simplify=TRUE))]


# 감성분석(bing은 단어사전의 일종)
start <- Sys.time()
nyt_sentence <- lapply(nyt_sentence, function(x){x[3] = get_sentiment(x[2], method="bing"); return(x)})
end <- Sys.time()
end-start   # 3mins

# 데이터 프레임으로 변경(뒤 코드들은 데이터프레임으로 처리하기가 간편, NA 데이터도 없음)
start <- Sys.time()
score <- as.data.frame(nyt_sentence)
score <- t(score)
end <- Sys.time()
end-start   # 55 secs

dimnames(score)[[1]] <- as.character(1:length(score)/dimnames(score)[[2]])
dimnames(score)[[2]] <- c("Date","Headline", "Score")

# 시간과 감성점수만 추출
score <- subset(score, select=c("Date","Score"))
write.csv(score,file=paste(save_folder,"/score.csv", sep=""))
