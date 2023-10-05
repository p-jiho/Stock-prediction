# NLP Preprocessing (NBC News)
# 수정할 점 : 생성한 함수가 너무 많아 코드가 복잡하고 깔끔하지 못함

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
import yfinance as yf
import datetime

year = list(map(str, list(range(2012,2023))))
month = list(map(str, list(range(1,13))))

for i in range(0,9):
    month[i] = "0"+month[i]

year_month = [0]*(12*(len(year)-1)+4)

k=0
for i in range(len(year)):
    for j in range(len(month)):
        if k < len(year_month):
            year_month[k] = year[i]+"_"+month[j]
            k = k+1
        else:
            k=k+1


def strip_split(list_low):
    list_low = list_low.strip()
    list_low = list_low.split("---")

    list_low[1] = pd.to_datetime(list_low[1].split(" UTC")[0], errors="ignore")
    list_low[2] = list_low[2] + " " + list_low[3]

    if list_low[1] == "":
        list_low = ""
    elif str(list_low[1]) < "2012-01-01" or str(list_low[1]) >= "2022-05-01":
        list_low = ""
    else:
        list_low = list_low[0:3]

    return list_low


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


dir = "/home/whfhrs3260/news_collection_data/"


def loading(data):
    f = open(dir + data + ".txt", "r", encoding="UTF-8")
    original_data = f.readlines()
    f.close()

    original_data = list(map(strip_split, original_data))
    original_data = [v for v in original_data if v]

    fullstop = re.compile(r'[,—"“”‘’\`\'-?:!;\\]')
    tit_txt_combination = [fullstop.sub(" ", str(original_data[i][2])) for i in range(len(original_data))]

    tit_txt_combination = list(map(word_tokenize, tit_txt_combination))

    ## ----------------------대문자 소문자 변환----------------------------------------------------
    tit_txt_combination = list(map(upper_to_lower_line, tit_txt_combination))  ## 첫번째 문자만 대문자인 경우 소문자로 변환

    ## ----------------------표제어 추출----------------------------------------------------
    tit_txt_combination = list(map(lemmatization_line, tit_txt_combination))

    ## ----------------------불용어 제거----------------------------------------------------
    tit_txt_combination = list(map(del_stopword, tit_txt_combination))

    return [original_data, tit_txt_combination]


def list_chunk(lst, num_cores):  ## num_cores에 맞게 df를 분리하는 함수  1,2,3,4 -> 코어가 2개인 경우 [1,2],[3,4]로 분류
    return [lst[i:i + num_cores] for i in range(0, len(lst), num_cores)]  ##   각 코어어 할당해주기 위해 필요


def type_conversion(df):  ## 한 코어가 실행한 n개의 데이터 프레임을 1개로 합치는 함수

    original_data = []
    tit_txt_combination = []

    for j in df:
        original_data_, tit_txt_combination_ = j
        original_data = original_data + original_data_
        tit_txt_combination = tit_txt_combination + tit_txt_combination_
    return [original_data, tit_txt_combination]


def parallel(df, func, num_cores):
    df_split = list_chunk(df, num_cores)  ## num_cores에 맞게 데이터 분리

    pool = Pool(num_cores)  ## process에 분배하여 함수 실행의 병렬처리를 도와줌

    def pool_map(df_split):
        df = pool.map(func, df_split)  ## 위에서 나눈 df_split을 각 process에 할당하여 실행
        return df

    df = list(map(pool_map, df_split))

    df = list(map(type_conversion, df))  ## 각각의 코어가 맡은 데이터프레임을 합침 즉, 1개의 코어가 2개의 데이터프레임을 실행했다면? 1개의 데이터프레임으로 합침

    ## 각 코어들이 만든 데이터 프레임도 합침
    original_data, tit_txt_combination = type_conversion(df)

    del df_split, df

    pool.close()
    pool.join()

    return original_data, tit_txt_combination


def main():
    df = year_month
    num_cores = multiprocessing.cpu_count()  # 28개 코어 : 3분
    original_data, tit_txt_combination = parallel(df, loading, num_cores)
    return original_data, tit_txt_combination

if __name__ == "__main__":
    start = datetime.datetime.now()
    original_data, tit_txt_combination = main()
    print(datetime.datetime.now() - start)


frequent_tit_txt = pd.Series(list(itertools.chain(*tit_txt_combination))).value_counts()
tit_txt_less_frequent_word = frequent_tit_txt[frequent_tit_txt<=10]
del frequent_tit_txt

def del_less_frequent_tit_txt(line):  ## 리스트의 한 line  ["a","b","c"]
    data = difference(line, list(set(line)& set(tit_txt_less_frequent_word.index)))   ## 불용어 사전과 동일하게 각 line과 불용어의 교집합 사전을 만들어 해당되는 값 제거
    return data

def work_func(tit_txt):
    tit_txt = list(map(del_less_frequent_tit_txt, tit_txt))
    return tit_txt

def main():
    num_cores = multiprocessing.cpu_count()  # 28개 코어 : 1시간
    df = tit_txt_combination
    df_split = np.array_split(df,num_cores)
    pool = Pool(num_cores)
    df = pool.map(work_func, df_split)
    pool.close()
    pool.join()

    return df

if __name__ == "__main__":
    start = datetime.datetime.now()
    tit_txt_combination = main()
    print(datetime.datetime.now()-start)

for i in range(len(tit_txt_combination)-1):
    i += 1
    tit_txt_combination[0].extend(tit_txt_combination[i])
tit_txt_combination = tit_txt_combination[0]

def date_cut_day(list_date_timestamp):
    list_date_timestamp[1] = datetime.datetime.strftime(list_date_timestamp[1], "%Y-%m-%d")
    return list_date_timestamp

original_data = list(map(date_cut_day, original_data))
tit_txt_combination = [[original_data[i][1]] + tit_txt_combination[i] for i in range(len(original_data))]
tit_txt_combination = sorted(tit_txt_combination, key=lambda date_plus_tit_txt: date_plus_tit_txt[0])

def from_combination_to_token(list_tit_txt_combination):
    list_tit_txt_combination = list_tit_txt_combination[1:len(list_tit_txt_combination)]
    return list_tit_txt_combination
tit_txt_token = list(map(from_combination_to_token, tit_txt_combination))

def word_to_sentence(lst):  ## 토큰화가 되어있는 tit, tit 데이터를 한 문장으로 만듦
    lst = " ".join(lst)     ## ex) the, and, me, bye => the and me bye
    return lst
tit_txt_sentence = list(map(word_to_sentence, tit_txt_token))

tit_txt_date = [[original_data[i][1]] for i in range(len(original_data))]
tit_txt_sentence = [tit_txt_date[i] + [tit_txt_sentence[i]] for i in range(len(tit_txt_sentence))]
analyzer = SentimentIntensityAnalyzer()
tit_txt_sentence = [tit_txt_sentence[i] + [analyzer.polarity_scores(tit_txt_sentence[i][1])["compound"]] for i in range(len(tit_txt_sentence))]
tit_txt_sentence = pd.DataFrame(tit_txt_sentence)
tit_txt_sentence.columns = ["Date","Title_Text", "Score"]
tit_txt_sentence = tit_txt_sentence[["Date","Score"]]

tit_txt_sentence.to_csv(dir+"data/Practice_data/price_data_score_10years.csv", index = False)