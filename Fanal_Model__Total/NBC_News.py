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
import func

# 연도와 월을 결합해 뽑을 연도와 열을 구성
# 2012년 1월 ~ 2022년 4월
year = list(map(str, list(range(2012, 2023))))
month = list(map(str, list(range(1, 13))))

for i in range(0, 9):
    month[i] = "0" + month[i]

year_month = [0] * (12 * (len(year) - 1) + 4)

k = 0
for i in range(len(year)):
    for j in range(len(month)):
        if k < len(year_month):
            year_month[k] = year[i] + "_" + month[j]
            k = k + 1
        else:
            k = k + 1

dir = "/home/whfhrs3260/news_collection_data/"


def loading(data):
    f = open(dir + data + ".txt", "r", encoding="UTF-8")
    original_data = f.readlines()
    f.close()

    original_data = list(map(func.strip_split, original_data))
    original_data = [v for v in original_data if v]

    fullstop = re.compile(r'[,—"“”‘’\`\'-?:!;\\]')
    tit_txt_combination = [fullstop.sub(" ", str(original_data[i][2])) for i in range(len(original_data))]

    tit_txt_combination = list(map(word_tokenize, tit_txt_combination))

    ## ----------------------대문자 소문자 변환----------------------------------------------------
    tit_txt_combination = list(map(func.upper_to_lower_line, tit_txt_combination))  ## 첫번째 문자만 대문자인 경우 소문자로 변환

    ## ----------------------표제어 추출----------------------------------------------------
    tit_txt_combination = list(map(func.lemmatization_line, tit_txt_combination))

    ## ----------------------불용어 제거----------------------------------------------------
    tit_txt_combination = list(map(func.del_stopword, tit_txt_combination))

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
    num_cores = multiprocessing.cpu_count()
    original_data, tit_txt_combination = parallel(df, loading, num_cores)
    return original_data, tit_txt_combination


if __name__ == "__main__":
    original_data, tit_txt_combination = main()

# 빈도수가 10 이하인 문자열은 제거

frequent_tit_txt = pd.Series(list(itertools.chain(*tit_txt_combination))).value_counts()
tit_txt_less_frequent_word = frequent_tit_txt[frequent_tit_txt <= 10]


def del_less_frequent_tit_txt(line):  ## 리스트의 한 line  ["a","b","c"]
    data = func.difference(line, list(
        set(line) & set(tit_txt_less_frequent_word.index)))  ## 불용어 사전과 동일하게 각 line과 불용어의 교집합 사전을 만들어 해당되는 값 제거
    return data


def work_func(tit_txt):
    tit_txt = list(map(del_less_frequent_tit_txt, tit_txt))
    return tit_txt


def main():
    num_cores = 28
    df = tit_txt_combination
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pool.map(work_func, df_split)
    pool.close()
    pool.join()

    return df


if __name__ == "__main__":
    tit_txt_combination = main()

# 차원 정리
for i in range(len(tit_txt_combination) - 1):
    i += 1
    tit_txt_combination[0].extend(tit_txt_combination[i])
tit_txt_combination = tit_txt_combination[0]

# date를 YYYY-MM-DD 형식으로 변경
original_data = list(map(func.nbc_date_cut_day, original_data))

# date와 뉴스 데이터 결합
tit_txt_combination = [[original_data[i][1]] + tit_txt_combination[i] for i in range(len(original_data))]

# token만 추출
tit_txt_token = list(map(func.from_combination_to_token, tit_txt_combination))

# token을 다시 문장으로 결합
tit_txt_sentence = list(map(func.word_to_sentence, tit_txt_token))

# date와 sentence 결합
tit_txt_date = [[original_data[i][1]] for i in range(len(original_data))]
tit_txt_sentence = [tit_txt_date[i] + [tit_txt_sentence[i]] for i in range(len(tit_txt_sentence))]

with open("data/NBC_News.txt", "w", encoding="UTF-8") as f:
    for line in tit_txt_sentence:
        f.write(line[0] + "%/%/%/%/%" + line[1] + "\n")