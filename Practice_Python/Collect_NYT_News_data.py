# Collect New York Times News Data
import json
import requests

import pandas as pd
import numpy as np
import regex as re

from tensorflow.keras.preprocessing.text import Tokenizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import multiprocessing
from multiprocessing import Pool

import warnings

warnings.filterwarnings('ignore')
import itertools

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import time


### 1번만 install
'''
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('vader_lexicon')
'''

year = list(map(str, list(range(2012, 2023))))
month = list(map(str, list(range(1, 13))))

year_month = []
for i in range(len(year)):
    for j in range(len(month)):
        year_month.append([year[i], month[j]])
year_month = year_month[0:124]


## 소문자 변환 - tit, txt에 적용(text, title을 토큰화한 리스트)
def upper_to_lower_line(line):  ## 한 리스트의 한 문장씩 불러와서 upper_to_lower_word 적용
    line = list(map(upper_to_lower_word, line))
    return line


def upper_to_lower_word(word):  ## string 형식, 한 문장의 한 단어씩 불러와서 대문자 변환   The -> the, THE -> THE
    word = re.sub("^[A-Z]{1}[a-z]*$", word.lower(), word)  ## 정규표현식에 맞는 단어만 lower 적용
    return word


## 표제어 추출 - tit, txt에 적용
def lemmatization_line(line):  ## 한 리스트의 한 문장씩 불러와서 lemmatization_line 적용
    line = list(map(lemmatization_word, line))
    return line


def lemmatization_word(data):  ## string 형식, 한 문장의 한 단어씩 불러와서 표제어 추출
    n = WordNetLemmatizer()
    data = n.lemmatize(data)
    return data


## 소문자 변환 - tit, txt에 적용(text, title을 토큰화한 리스트)
def upper_to_lower_line(line):  ## 한 리스트의 한 문장씩 불러와서 upper_to_lower_word 적용
    line = list(map(upper_to_lower_word, line))
    return line


def upper_to_lower_word(word):  ## string 형식, 한 문장의 한 단어씩 불러와서 대문자 변환   The -> the, THE -> THE
    word = re.sub("^[A-Z]{1}[a-z]*$", word.lower(), word)  ## 정규표현식에 맞는 단어만 lower 적용
    return word


## 표제어 추출 - tit, txt에 적용
def lemmatization_line(line):  ## 한 리스트의 한 문장씩 불러와서 lemmatization_line 적용
    line = list(map(lemmatization_word, line))
    return line


def lemmatization_word(data):  ## string 형식, 한 문장의 한 단어씩 불러와서 표제어 추출
    n = WordNetLemmatizer()
    data = n.lemmatize(data)
    return data


## 불용어 제거 - tit, txt에 적용
def del_stopword(line):  ## 한 리스트의 한 문장씩 불러와서 dir_stopword_produce 적용
    dir_stop_words = stopwords.words('english')  ## 불용어 사전

    line_stopwords_intersection = list(set(line) & set(dir_stop_words))  ## 각 문장과 불용어 사전에 동시에 있는 단어 추출

    # 각 문장마다 불용어 사전과 교집합인 사전 생성

    # 각 문장마다 교집합 사전에 해당하지 않는 값만 추출
    line = difference(line, line_stopwords_intersection)

    return line


def difference(line, line_stopwords_intersection):  ## 각 문장, 각 문장과 불용어 사전의 교집합 입력
    line = [i for i in line if i not in line_stopwords_intersection]  ## 불용어 사전에 해당하지 않는 단어만 추출
    return line


def nyt_preprocess(year_month):
    try:
        key = "G7AS6LA9s8ElEiMAMNYKoWgAFFq4ouAJ"
        year = year_month[0]
        month = year_month[1]
        nyt_base_url = "https://api.nytimes.com/svc/archive/v1/" + str(year) + "/" + str(month) + ".json?api-key=" + key
        nyt_base_url_result = requests.get(nyt_base_url)
        nyt_base_url_text = nyt_base_url_result.text
        nyt_news = json.loads(nyt_base_url_text)
    except Exception:
        return None
    finally:
        time.sleep(10)

    nyt_title = list(map(lambda x: x["headline"]["main"], nyt_news["response"]["docs"]))
    nyt_date = list(map(lambda x: x["pub_date"], nyt_news["response"]["docs"]))

    nyt_headline_date = []
    for i in range(len(nyt_title)):
        nyt_headline_date.append([nyt_title[i], nyt_date[i]])

    nyt_headline_date = list(
        map(lambda x: "" if x[1] == "" else ("" if str(x[1]) < "2012-01-01" or str(x[1]) >= "2022-05-01" else x),
            nyt_headline_date))
    nyt_headline_date = [v for v in nyt_headline_date if v]

    fullstop = re.compile(r'[,—"“”‘’\`\'-?:!;\\]')
    nyt_headline = [fullstop.sub(" ", str(nyt_headline_date[i][0])) for i in range(len(nyt_headline_date))]

    nyt_headline = list(map(word_tokenize, nyt_headline))

    ## ----------------------대문자 소문자 변환----------------------------------------------------
    nyt_headline = list(map(upper_to_lower_line, nyt_headline))  ## 첫번째 문자만 대문자인 경우 소문자로 변환

    ## ----------------------표제어 추출----------------------------------------------------
    nyt_headline = list(map(lemmatization_line, nyt_headline))

    ## ----------------------불용어 제거----------------------------------------------------
    nyt_headline = list(map(del_stopword, nyt_headline))

    return [nyt_headline_date, nyt_headline]


nyt_collection_result = list(map(lambda x: nyt_preprocess(x), year_month[0:1]))

nyt_headline_date = list(map(lambda x: x[0], nyt_collection_result))
nyt_headline_token = list(map(lambda x: x[1], nyt_collection_result))

nyt_headline_date = list(itertools.chain.from_iterable(nyt_headline_date))
nyt_headline_token = list(itertools.chain.from_iterable(nyt_headline_token))

frequent_headline = pd.Series(list(itertools.chain(*nyt_headline_token))).value_counts()
headline_less_frequent_word = frequent_headline[frequent_headline>=10]
del frequent_headline

def del_less_frequent_headline(line):  ## 리스트의 한 line  ["a","b","c"]
    data = difference(line, list(set(line)& set(headline_less_frequent_word)))   ## 불용어 사전과 동일하게 각 line과 불용어의 교집합 사전을 만들어 해당되는 값 제거
    return data

def del_less_headline_fun(headline):
    headline = list(map(del_less_frequent_headline, headline))
    return headline

def main():
    num_cores = multiprocessing.cpu_count()   # 28개 코어 : 1분 15초
    lst = nyt_headline_token
    lst_split = np.array_split(lst,num_cores)
    pool = Pool(num_cores)
    lst = pool.map(del_less_headline_fun, lst_split)
    pool.close()
    pool.join()

    return lst

if __name__ == "__main__":
    start = datetime.datetime.now()
    nyt_headline_token = main()
    print(datetime.datetime.now()-start)



for i in range(len(nyt_headline_token)-1):
    i += 1
    nyt_headline_token[0].extend(nyt_headline_token[i])
nyt_headline_token = nyt_headline_token[0]

def date_cut_day(list_date_timestamp):
    list_date_timestamp[1] = pd.to_datetime(list_date_timestamp[1])
    list_date_timestamp[1] = datetime.datetime.strftime(list_date_timestamp[1], "%Y-%m-%d")
    return list_date_timestamp

nyt_headline_date = list(map(date_cut_day, nyt_headline_date))

nyt_headline_date_token = [[nyt_headline_date[i][1]] + nyt_headline_token[i] for i in range(len(nyt_headline_date))]
nyt_headline_date_token = sorted(nyt_headline_date_token, key=lambda date_plus_tit_txt: date_plus_tit_txt[0])

def from_combination_to_token(list_tit_txt_combination):
    list_tit_txt_combination = list_tit_txt_combination[1:len(list_tit_txt_combination)]
    return list_tit_txt_combination
nyt_headline_token = list(map(from_combination_to_token, nyt_headline_date_token))

def word_to_sentence(lst):  ## 토큰화가 되어있는 tit, tit 데이터를 한 문장으로 만듦
    lst = " ".join(lst)     ## ex) the, and, me, bye => the and me bye
    return lst

nyt_headline_sentence = list(map(word_to_sentence, nyt_headline_token))

nyt_date = [[nyt_headline_date[i][1]] for i in range(len(nyt_headline_date))]

nyt_headline_sentence = [nyt_date[i] + [nyt_headline_sentence[i]] for i in range(len(nyt_headline_sentence))]

analyzer = SentimentIntensityAnalyzer()
nyt_headline_sentence = [nyt_headline_sentence[i] + [analyzer.polarity_scores(nyt_headline_sentence[i][1])["compound"]] for i in range(len(nyt_headline_sentence))]

nyt_headline_sentence = pd.DataFrame(nyt_headline_sentence)
nyt_headline_sentence.columns = ["Date","Headline", "Score"]
nyt_headline_sentence = nyt_headline_sentence[["Date","Score"]]

nyt_headline_sentence.to_csv("../data/NYT_News/nyt_headline_score.csv", index = False)