# 첫번째. NBC 뉴스 전처리
import pandas as pd
import numpy as np
import regex as re

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import multiprocessing
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')
import itertools

year = list(map(str, list(range(2017,2023))))
month = list(map(str, list(range(1,13))))

for i in range(0,9):
    month[i] = "0"+month[i]

year_month = [0]*(12*5+4)

k=0
for i in range(len(year)):
    for j in range(len(month)):
        if k < len(year_month):
            year_month[k] = year[i]+"_"+month[j]
            k = k+1
        else:
            k=k+1


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


dir = "/home/whfhrs3260/data/"


def loading(data):
    ## ----------------------데이터 파일 불러오기--------------------------------------------------
    f = open(dir + str(data) + ".txt", "r", encoding="UTF-8")
    datas = f.readlines()
    f.close()

    ## ----------------------데이터 저장--------------------------------------------------
    n = len(datas)
    m = 0
    original_data = [0] * n
    for line in datas:
        original_data[m] = line.strip()  ## 한줄씩 읽기
        m += 1

    for k in range(n):
        original_data[k] = original_data[k].split(" --- ")  ## 구분자를 이용해 구분

    original_data = pd.DataFrame(original_data)
    original_data.columns = ["category", "date", "title", "text", "author"]  # specify the column name
    original_data.index = list(range(len(original_data)))

    ## ----------------------데이터 구조 변경----------------------------------------------------
    data_title_indx = np.array(np.where(original_data.iloc[:, 3].isnull())[0])
    data_title = original_data.iloc[data_title_indx, :]
    data_text_indx = np.array(np.where(original_data.iloc[:, 4].isnull())[0])
    data_text = original_data.iloc[np.setdiff1d(data_text_indx, data_title_indx), :]

    # case 1. title만 있는 경우 변경
    data_1_indx = np.intersect1d(data_title_indx, original_data[original_data["title"] == " ---"].index)
    data_1 = original_data.iloc[data_1_indx, :]  ## case 1에 맞는 데이터
    data_1.title = data_1.date
    data_1.loc[:, ["category", "date"]] = None
    original_data.iloc[data_1_indx, :] = data_1

    # case 2. title과 text만 있는 경우 변경
    data_2_indx = np.setdiff1d(data_title_indx, data_1_indx)
    data_2 = original_data.iloc[data_2_indx, :]
    data_2.text = data_2.title
    data_2.title = data_2.date
    data_2.loc[:, ["category", "date"]] = None
    original_data.iloc[data_2_indx, :] = data_2

    # case 3. title, text, author만 있는 경우
    data_3_indx = np.intersect1d(data_text_indx, original_data[original_data["category"] == "--- "].index)
    data_3 = original_data.iloc[data_3_indx, :]
    data_3.author = data_3.text
    data_3.text = data_3.title
    data_3.title = data_3.date
    data_3.loc[:, ["category", "date"]] = None
    original_data.iloc[data_3_indx, :] = data_3

    # 기사 최초 작성일만 추출
    original_data.date = original_data.date.str.split("/").str[0]

    ## ----------------------불필요한 데이터 제거----------------------------------------------------
    date_null_ind = np.array(np.where(original_data.iloc[:, [1]].isnull())[0])
    date_black_ind = np.array(np.where(original_data.date == "")[0])

    del_date_ind = np.concatenate((date_null_ind, date_black_ind))  # date 열이 None인 행의 인덱스 추출
    original_data = original_data.drop(del_date_ind)  # 인덱스에 해당하는 행 삭제 즉, date 열이 None인 행 삭제

    ## ----------------------데이터 시간 형식 변경----------------------------------------------------
    ## 원래 date : Jan. 1, 2021, 12:23 AM UTC
    ## 가공된 date : 2021-01-01 00:23:00
    original_data.date = original_data.date.apply(
        lambda x: x.strip(" UTC\xa0"))  ## data의 date 열에서 UTC\xa0이나 UTC에 해당하는 부분 삭제
    original_data.date = original_data.date.apply(lambda x: x.strip(" UTC"))

    original_data.date = original_data.date.apply(lambda x: pd.to_datetime(x, errors="ignore"))  ## datetime 형식으로 변경

    # 기준 이전의 날짜는 자르기
    original_data = original_data[original_data.date >= "2012-01-01"]
    original_data = original_data[original_data.date < "2022-05-01"]

    ## ---------------------text와 title 결합--------------------
    original_data["title_text"] = original_data["title"] + " " + original_data["text"]

    ## ----------------------구두점 제거----------------------------------------------------
    fullstop = re.compile(r'[,—"“”‘’\'-?:!;\\]')
    original_data.title_text = original_data.title_text.apply(lambda x: fullstop.sub(" ", x))  ## title에서 제거

    ## ----------------------토큰화----------------------------------------------------
    tit_txt_combination = list(map(word_tokenize, original_data.title_text))

    ## ----------------------대문자 소문자 변환----------------------------------------------------
    tit_txt_combination = list(map(upper_to_lower_line, tit_txt_combination))  ## 첫번째 문자만 대문자인 경우 소문자로 변환

    ## ----------------------표제어 추출----------------------------------------------------
    tit_txt_combination = list(map(lemmatization_line, tit_txt_combination))

    ## ----------------------불용어 제거----------------------------------------------------
    tit_txt_combination = list(map(del_stopword, tit_txt_combination))

    ## ----------------------데이터 프레임으로 형식 변환----------------------------------------------------
    tit_txt_combination = pd.DataFrame(tit_txt_combination)  # 데이터를 정리하기 위해서 변환이 필요

    return [original_data, tit_txt_combination]


def list_chunk(lst, num_cores):  ## num_cores에 맞게 df를 분리하는 함수  1,2,3,4 -> 코어가 2개인 경우 [1,2],[3,4]로 분류
    return [lst[i:i + num_cores] for i in range(0, len(lst), num_cores)]  ##   각 코어어 할당해주기 위해 필요


def type_conversion(df):  ## 한 코어가 실행한 n개의 데이터 프레임을 1개로 합치는 함수

    original_data = pd.DataFrame()
    tit_txt_combination = pd.DataFrame()

    for j in df:
        original_data_, tit_txt_combination_ = j
        original_data = pd.concat([original_data, original_data_])
        tit_txt_combination = pd.concat([tit_txt_combination, tit_txt_combination_])
    return [original_data, tit_txt_combination]


def parallel(df, func, num_cores):
    df_split = list_chunk(df, num_cores)  ## num_cores에 맞게 데이터 분리

    pool = Pool(num_cores)  ## process에 분배하여 함수 실행의 병렬처리를 도와줌

    def pool_map(df_split):
        df = pool.map(loading, df_split)  ## 위에서 나눈 df_split을 각 process에 할당하여 실행
        return df

    df = list(map(pool_map, df_split))

    df = list(map(type_conversion, df))  ## 각각의 코어가 맡은 데이터프레임을 합침 즉, 1개의 코어가 2개의 데이터프레임을 실행했다면? 1개의 데이터프레임으로 합침

    ## 각 코어들이 만든 데이터 프레임도 합침
    original_data, tit_txt_combination = type_conversion(df)

    pool.close()
    pool.join()

    return original_data, tit_txt_combination


def main():
    df = year_month
    num_cores = multiprocessing.cpu_count()  # 28개 코어 : 3분
    original_data, tit_txt_combination = parallel(df, loading, num_cores)
    return original_data, tit_txt_combination


if __name__ == "__main__":
    original_data, tit_txt_combination = main()

tit_txt_combination = tit_txt_combination.where(pd.notnull(tit_txt_combination), None)

## 빈도수 기반 사전 생성
def del_list_None(lst):
    lst = list(filter(None, lst))
    return lst

tit_txt_combination = tit_txt_combination.values.tolist()  ## 데이터프레임 -> list
tit_txt_combination = list(map(del_list_None, tit_txt_combination))   ## 변환하면서 생긴 None 삭제

frequent_tit_txt = pd.DataFrame(pd.Series(list(itertools.chain(*tit_txt_combination))).value_counts())   ## tit에 있는 단어들의 빈도수 추출
tit_txt_less_frequent_word = list(frequent_tit_txt[4 >= frequent_tit_txt[0]].index)     ## 빈도가 4 이하인 단어를 추출

from multiprocessing import Pool

def del_less_frequent_tit_txt(line):  ## 리스트의 한 line  ["a","b","c"]
    line_frequent_intersection = list(set(line)& set(tit_txt_less_frequent_word))   ## 불용어 사전과 동일하게 각 line과 불용어의 교집합 사전을 만들어 해당되는 값 제거
    data = difference(line, line_frequent_intersection)
    return data

def work_func(tit_txt_combination):
    tit_txt_combination = list(map(del_list_None, tit_txt_combination))
    tit_txt_combination = list(map(del_less_frequent_tit_txt, tit_txt_combination))
    return tit_txt_combination

def main():
    num_cores = multiprocessing.cpu_count()  # 28개 코어 : 4분
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

tit_txt_combination = pd.DataFrame(tit_txt_combination)

# 용량이 커 csv 파일로 저장하지 못하고 pickle을 사용
dir = "../../../../"
original_data.to_pickle(dir+"data/Practice_data/original_data_4years.pkl")
tit_txt_combination.to_pickle(dir+"data/Practice_data/tit_txt_combination_4years.pkl")
