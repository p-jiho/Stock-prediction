#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import datetime


# In[2]:


def strip_split(list_low):
    list_low = list_low.strip()
    list_low = list_low.split("---")
    
    list_low[1] = pd.to_datetime(list_low[1].split(" UTC")[0], errors="ignore")
    list_low[2] = list_low[2] + " " + list_low[3]
        
    if list_low[1]=="":
        list_low = ""
    elif str(list_low[1])<"2012-01-01" or str(list_low[1])>="2022-05-01":
        list_low = ""
    else: 
        list_low = list_low[0:3] 
    
    
    
    return list_low



## 소문자 변환 - tit, txt에 적용(text, title을 토큰화한 리스트)
def upper_to_lower_line(line):  ## 한 리스트의 한 문장씩 불러와서 upper_to_lower_word 적용
    line = list(map(upper_to_lower_word,line))
    return line

def upper_to_lower_word(word):  ## string 형식, 한 문장의 한 단어씩 불러와서 대문자 변환   The -> the, THE -> THE
    word = re.sub("^[A-Z]{1}[a-z]*$",word.lower(),word)  ## 정규표현식에 맞는 단어만 lower 적용
    return word

## 표제어 추출 - tit, txt에 적용
def lemmatization_line(line):   ## 한 리스트의 한 문장씩 불러와서 lemmatization_line 적용
    line = list(map(lemmatization_word, line))
    return line

def lemmatization_word(data):  ## string 형식, 한 문장의 한 단어씩 불러와서 표제어 추출
    n=WordNetLemmatizer()
    data = n.lemmatize(data)
    return data

## 소문자 변환 - tit, txt에 적용(text, title을 토큰화한 리스트)
def upper_to_lower_line(line):  ## 한 리스트의 한 문장씩 불러와서 upper_to_lower_word 적용
    line = list(map(upper_to_lower_word,line))
    return line

def upper_to_lower_word(word):  ## string 형식, 한 문장의 한 단어씩 불러와서 대문자 변환   The -> the, THE -> THE
    word = re.sub("^[A-Z]{1}[a-z]*$",word.lower(),word)  ## 정규표현식에 맞는 단어만 lower 적용
    return word



## 불용어 제거 - tit, txt에 적용
def del_stopword(line):              ## 한 리스트의 한 문장씩 불러와서 dir_stopword_produce 적용
    dir_stop_words = stopwords.words('english')  ## 불용어 사전
    
    line_stopwords_intersection = list(set(line)& set(dir_stop_words))   ## 각 문장과 불용어 사전에 동시에 있는 단어 추출
    
    # 각 문장마다 불용어 사전과 교집합인 사전 생성
    
    # 각 문장마다 교집합 사전에 해당하지 않는 값만 추출
    line = difference(line, line_stopwords_intersection)  
    
    return line


def difference(line, line_stopwords_intersection):      ## 각 문장, 각 문장과 불용어 사전의 교집합 입력
    line = [i for i in line if i not in line_stopwords_intersection]  ## 불용어 사전에 해당하지 않는 단어만 추출
    return line


def nbc_date_cut_day(list_date_timestamp):
    list_date_timestamp[1] = datetime.datetime.strftime(list_date_timestamp[1], "%Y-%m-%d")
    return list_date_timestamp

def nyt_date_cut_day(list_date_timestamp):
    list_date_timestamp[1] = pd.to_datetime(list_date_timestamp[1])
    list_date_timestamp[1] = datetime.datetime.strftime(list_date_timestamp[1], "%Y-%m-%d")
    return list_date_timestamp

def from_combination_to_token(list_tit_txt_combination):
    list_tit_txt_combination = list_tit_txt_combination[1:len(list_tit_txt_combination)]
    return list_tit_txt_combination

def word_to_sentence(lst):  ## 토큰화가 되어있는 tit, tit 데이터를 한 문장으로 만듦
    lst = " ".join(lst)     ## ex) the, and, me, bye => the and me bye
    return lst


# In[ ]:




