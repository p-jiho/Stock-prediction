library(rvest)
library(purrr) # map 함수

year <- sapply(c(2012:2022), function(x){as.character(x)})
# https://rbasall.tistory.com/151
month_number <- sapply(c(1:12), function(x){if(x>=10){x}else{paste("0",x,sep="")}})
month <- data.frame("month_name" = c(tolower(month.name)),"month_number" = c(month_number))

map(year, function(years){
  apply(month,1, function(month){
    main_url <- paste("https://www.nbcnews.com/archive/articles/",years,"/",month[1], sep="")
    html_news <- read_html(main_url,encoding='UTF-8')
    
    page <- html_news %>%
      html_nodes("nav > div > a") %>%
      html_text()
    page <- c(1, page)
    
    news <- list()
    i=0
    news <- map(page, function(page){
      if(page==1){
        html_href <- read_html(main_url)
      }else{html_href <- read_html(paste(main_url,"/",page,sep=""))}
      
      print(page)
      
      href <- html_href %>% 
        html_nodes("body") %>%
        html_nodes("div.Layout")%>%
        html_nodes("main") %>%
        html_nodes('a') %>%
        html_attr('href')

      news <- map(href, function(href){
        try({
        print(href)
        html_news <- read_html(href, encoding = "UTF-8")
        catagory <- html_news %>%
          html_nodes("header > aside > div > a > span") %>%
          html_text()
        
        date <- html_news %>%
          html_nodes("time")%>%
          html_text()
        
        title <- html_news %>%
          html_nodes("header > div > h1")%>%
          html_text()
        
        text <- html_news %>%
          html_nodes("div > div > div > div.article-body__content > p")%>%
          html_text() %>%
          paste(collapse=" ")
        
        reporter <- c(
          html_news %>%
          html_nodes("div.article-inline-byline > span > a") %>%
          html_text(),
          html_news %>%
            html_nodes("section > div.article-inline-byline > span") %>%
            html_text()
        )
        news <- append(news, paste(catagory, date, title, text, reporter,sep = "////----////----////"))
        }, silent = FALSE)
        return(news)
      })
    })
    news <- paste(unlist(as.vector(news)), sep="/n")
    write.table(news,file = paste("./R_code/news_data/", years,"_",month[2],".txt", sep=""), col.names = FALSE, row.names = FALSE)
  })
})
