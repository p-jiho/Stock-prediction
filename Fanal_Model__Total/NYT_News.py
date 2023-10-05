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
import func

year = list(map(str, list(range(2012, 2023))))
month = list(map(str, list(range(1, 13))))

year_month = []
for i in range(len(year)):
    for j in range(len(month)):
        year_month.append([year[i], month[j]])
year_month = year_month[0:124]


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
    nyt_headline = list(map(func.upper_to_lower_line, nyt_headline))  ## 첫번째 문자만 대문자인 경우 소문자로 변환

    ## ----------------------표제어 추출----------------------------------------------------
    nyt_headline = list(map(func.lemmatization_line, nyt_headline))

    ## ----------------------불용어 제거----------------------------------------------------
    nyt_headline = list(map(func.del_stopword, nyt_headline))

    return [nyt_headline_date, nyt_headline]


nyt_collection_result = list(map(lambda x: nyt_preprocess(x), year_month))

nyt_headline_date = list(map(lambda x: x[0], nyt_collection_result))
nyt_headline_token = list(map(lambda x: x[1], nyt_collection_result))

nyt_headline_date = list(itertools.chain.from_iterable(nyt_headline_date))
nyt_headline_token = list(itertools.chain.from_iterable(nyt_headline_token))

frequent_headline = pd.Series(list(itertools.chain(*nyt_headline_token))).value_counts()
headline_less_frequent_word = frequent_headline[frequent_headline<=10]

def del_less_frequent_headline(line):  ## 리스트의 한 line  ["a","b","c"]
    data = func.difference(line, list(set(line)& set(headline_less_frequent_word.index)))   ## 불용어 사전과 동일하게 각 line과 불용어의 교집합 사전을 만들어 해당되는 값 제거
    return data

def del_less_headline_fun(headline):
    headline = list(map(del_less_frequent_headline, headline))
    return headline

def main():
    num_cores = multiprocessing.cpu_count()
    lst = nyt_headline_token
    lst_split = np.array_split(lst,num_cores)
    pool = Pool(num_cores)
    lst = pool.map(del_less_headline_fun, lst_split)
    pool.close()
    pool.join()

    return lst

if __name__ == "__main__":
    nyt_headline_token = main()

# 차원 정리
for i in range(len(nyt_headline_token)-1):
    i += 1
    nyt_headline_token[0].extend(nyt_headline_token[i])
nyt_headline_token = nyt_headline_token[0]

# date 형식 YYYY-MM-DD
nyt_headline_date = list(map(func.nyt_date_cut_day, nyt_headline_date))

# token + date
nyt_headline_date_token = [[nyt_headline_date[i][1]] + nyt_headline_token[i] for i in range(len(nyt_headline_date))]

# 시간 순으로 sorted
nyt_headline_date_token = sorted(nyt_headline_date_token, key=lambda date_plus_tit_txt: date_plus_tit_txt[0])

# 정렬된 데이터를 다시 토큰만 추출
nyt_headline_token = list(map(func.from_combination_to_token, nyt_headline_date_token))

# 문장으로 변환
nyt_headline_sentence = list(map(func.word_to_sentence, nyt_headline_token))

# 날짜 추출하여 문장과 결합
nyt_date = [[nyt_headline_date[i][1]] for i in range(len(nyt_headline_date))]
nyt_headline_sentence = [nyt_date[i] + [nyt_headline_sentence[i]] for i in range(len(nyt_headline_sentence))]

with open("data/NYT_News.txt", "w", encoding="UTF-8") as f:
    for line in nyt_headline_sentence:
        f.write(line[0] + "%/%/%/%/%" + line[1] + "\n")