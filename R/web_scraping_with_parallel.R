########################## nbc hrefs collection ###########################



### collecting url of nbc news for a month ###
### ---------------------------------------------------- ###
# Receive year and month to create url.
# Url collected a month's news.
# After importing the html of the url, collect the url of the collected news.
# Collect and combine multi-page news
### ---------------------------------------------------- ###
# receive year and month.
nbc_href_collection <- function(year, month){                                                   
  # base url of nbc news
  nbc_base_url <- "https://www.nbcnews.com/archive/articles/"                                   
  
  # create url corresponding to year and month entered
  month_url <- paste(nbc_base_url, year, "/", tolower(month.name[as.integer(month)]), sep = "") 
  
  # collect html of corresponding url
  html_href <- read_html(month_url, encoding = 'UTF-8')                                         

  # collect url of news on the first page of the year and month entered
  hrefs_one_page <- html_href %>%                         
    html_nodes("body > div.Layout > main > a") %>%
    html_attr('href')         
  
  # number of pages corresponding to year and month entered
  pages <- html_news %>%                             
    html_nodes("nav > div > a") %>%
    html_text()

  # looping for more than two pages
  hrefs = sapply(pages, function(page){  

    # create url to indicate two or more pages
    html_href <- read_html(paste(month_url, "/", page, sep = ""))   

    # collect url of news on the page of the year and month entered
    hrefs_one_page <- html_href %>%                           
      html_nodes("body > div.Layout > main > a") %>%
      html_attr('href')    
  
    return(hrefs_one_page)
  })

  # organize url on two or more pages as vector
  hrefs = as.vector(unlist(hrefs))         

  # combine url on the first page and after page two
  hrefs = c(hrefs_one_page, hrefs)         
  
  return(hrefs)
}

# year and month designation
year <- 2012  
month <- "01"  

# collect url on all pages using the above function
nbc_hrefs = nbc_href_collection(year, month)    

# save collected url
write.table(                           
  nbc_hrefs,
  file = paste("./R_code/news_hrefs/", year, "_", month, ".txt", sep =
                 ""),
  col.names = FALSE,
  row.names = FALSE
)

################################# nbc news collection #####################
require("parallel")
require("httr")

### collecting catagory, date, title, text and reporter of nbc news for a month ###
### ---------------------------------------------------- ###
# Collect html by calling up url of collected and stored news
# Collect a total of 5 variables
# Use delimiters to combine and store
### ---------------------------------------------------- ###
# receive url from news
nbc_news_collection <- function(nbc_href) {
    try({
      
      # collect html of news url
      html_news <-read_html(nbc_href, encoding = 'UTF-8')   
      
      # collect catagory of news
      catagory <- html_news %>%    
        html_nodes("header > aside > div > a > span") %>%
        html_text()
      
      # collect date of news
      date <- html_news %>%     
        html_nodes("time") %>%
        html_text()
      
      # collect title of news
      title <- html_news %>%    
        html_nodes("header > div > h1") %>%
        html_text()
      
      # collect text of news
      text <- html_news %>%     
        html_nodes("div > div > div > div.article-body__content > p") %>%
        html_text() %>%
        paste(collapse = " ")
      
      # collect reporter of news
      reporter <- c(        
        
        # the first case
        html_news %>%       
          html_nodes("div.article-inline-byline > span > a") %>%
          html_text(),
        
        # the second case
        html_news %>%      
          html_nodes("section > div.article-inline-byline > span") %>%
          html_text()
      )
      
      # combine values collected using delimiters
      news <- paste(catagory, date, title, text, reporter, sep = "////----////----////")      
      
      return(news)
      
    }, silent = FALSE) # passing by without producing results when an error occurs
}

year <- 2012
month <- "01"

# bring up url collected for a month
nbc_hrefs <- read.table(paste("./R_code/news_hrefs/", year, "_", month, ".txt", sep =""))  

# convert to vector
nbc_hrefs <- as.vector(nbc_hrefs)[[1]]            

# number of core
numcore <- 6        

### paraller ###
clus <- makeCluster(numcore, type = "PSOCK")  

clusterEvalQ(clus,require('xml2'))
clusterEvalQ(clus,require('dplyr'))
clusterEvalQ(clus,require('rvest'))
clusterEvalQ(clus,require('httr'))

# collect url's news in parallel
nbc_news_data <- parLapply(cl=clus, X = nbc_hrefs, fun = nbc_news_collection)   
## If there are six cores, they are optimized in terms of time.
## numcore = 2 ===> 15 mins
## numcore = 3 ===> 8 mins
## numcore = 4 ===> 6 mins
## numcore = 5 ===> 4.8 mins
## numcore = 6 ===> 4.1 mins


stopCluster(clus)

# convert to vector
nbc_news_data <- unlist(nbc_news_data)   

# save collected nbc news data
write.table(             
  nbc_news_data,
  file = paste("./R_code/news_data/", year, "_", month, ".txt", sep =
                 ""),
  col.names = FALSE,
  row.names = FALSE
)
