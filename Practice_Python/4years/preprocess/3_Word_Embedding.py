# 세번째, 워드임베딩 시도 -> 결과적으로 사용하지 않음(감성분석으로 방향 선정)
import pandas as pd
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer

import smart_open
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from gensim.models import FastText

from sentence_transformers import SentenceTransformer

import copy

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

dir = "../../../../"
original_news_dir = dir + "data/Practice_data/original_data.pkl"
tit_txt_dir = dir + "data/Practice_data/tit_txt_combination_4years.pkl"

original_news = pd.read_pickle(original_news_dir)
tit_txt_combination = pd.read_pickle(tit_txt_dir)

original_news.date = original_news.date.apply(lambda x: pd.to_datetime(x, errors="ignore"))

## 시간을 날짜 단위로 짜름
def date_until_day(date_timestamp):  ## 2021-01-01 00:23:00 => 2021-01-01
    date_timestamp = datetime.datetime.strftime(date_timestamp, "%Y-%m-%d")
    return date_timestamp


news_day_standard = copy.deepcopy(original_news)     ## original data를 value만 copy

news_day_standard.date = news_day_standard.date.apply(lambda x: date_until_day(x))  ## 시간을 시 단위로 짜름

tit_txt_combination = tit_txt_combination.where(pd.notnull(tit_txt_combination), None) ## NaN -> None으로 변경(데이터 처리를 편리하게 하기 위해)
tit_txt_date_day = pd.concat([news_day_standard.date,tit_txt_combination],axis=1) ## 데이터에 date 변수 붙여주기
tit_txt_date_day = tit_txt_date_day.sort_values(by='date') ## 시간 순으로 정렬
tit_txt_date_day = tit_txt_date_day.reset_index(drop=True) ## 새로 정렬된 데이터의 인덱스를 순서대로 재설정  # index 100 51 21 40 ... => 1 2 3 4 5 ...
tit_txt_date_day = tit_txt_date_day.values.tolist() # 데이터프레임을 list로 변환
## 데이터 프레임을 list로 변환하는 과정에서 포함된 None을 삭제
def del_list_None(lst):
    lst = list(filter(None, lst))
    return lst

tit_txt_date_day = list(map(del_list_None, tit_txt_date_day ))

# 같은 시간인 경우 데이터를 합침 ex) 1월 1일 10시~10시 59분 사이의 기사는 전부 합침

## 알고리즘 설명
## basic_date 즉, 기준이 되는 date가 1월 1일 10시라고 가정
## 현재 loop의 date가 1월 1일 10시라면 원래 있던 1월 1일 10시 데이터와 새로운 데이터를 결합
## 현재 loop의 date가 1월 1일 10시가 아닌 1월 1일 11시로 새로운 date가 출현했다면 basic_date 즉, 기준 데이터를 1월 1일 11시로 변경하고 새로운 리스트를 생성
## 이를 반복해 같은 시간인 뉴스끼리 결합되고 다른 시간인 경우는 다른 list로 분리
## 결국, 1시간 당 1개의 리스트만 생성

# 같은 날짜인 경우 데이터를 합침 ex) 1월 1일 00시 00분 ~ 23시 59분 사이의 기사는 전부 합침
length = -1
tit_txt_combination_date_day = []

basic_date = 0  ## 기준이 되는 date
for i in range(len(tit_txt_date_day)):  ## 전체를 한번씩 돈다
    new_date = tit_txt_date_day[i][0]  # new date는 현재 loop의 date
    if basic_date == new_date:  # 현재 loop의 date가 기준 date와 같으면 실행
        tit_txt_combination_date_day[length] = tit_txt_combination_date_day[length] + tit_txt_date_day[i][1:(
            len(tit_txt_date_day[i]))]  ## 앞의 데이터에 새로운 데이터를 결합
    else:  # 현재 loop의 date가 기준 date와 다르면 실행 즉, 새로운 시간이 나타나면 실행
        length += 1  # 길이가 한개 늘어남
        tit_txt_combination_date_day.append(tit_txt_date_day[i][1:(len(tit_txt_date_day[i]))])  # 새로운 list 데이터 추가
        basic_date = tit_txt_date_day[i][0]  # 기준 date를 새로운 date로 변경


## -------------------------------------- 1. Bag of Words 생성 ---------------
## BOW를 만드는 TfidfVectorizer 함수를 사용하기 위해서는 토큰화가 되지 않은 문장 데이터가 필요
## => 토큰화가 되어있는 tit, txt 데이터를 한 문장으로 만들어 list로 구성
## Bag of Words : 한 문서에 있는 단어들의 집합, 없는 단어면 0, 있는 단어면 갯수만큼 n => ex) 0 0 0 1 0 2 3 0 2
## tf-idf : 단어의 빈도와 역 문서 빈도를 활용하여 중요한 단어에 가중치를 두는 방식
## 함수에 적용할 데이터 구조 수정
def word_to_sentence(lst):  ## 토큰화가 되어있는 tit, tit 데이터를 한 문장으로 만듦
    lst = " ".join(lst)     ## ex) the, and, me, bye => the and me bye
    return lst

## 각 문장마다 수행, 결과 ex) ["the and me bye", ...,"i do not me"]
tit_txt_sentence_day = list(map(word_to_sentence, tit_txt_combination_date_day))

## BOW를 만듦 + tf-idf 적용
bow_tfidf_tit_txt_day = TfidfVectorizer().fit(tit_txt_sentence_day)

## BOW에서 각 요소의 이름 추출
bow_tfidf_vocab_tit_txt_day = bow_tfidf_tit_txt_day.get_feature_names_out()

# DataFrame 형식으로 BOW + tf-idf 데이터 구성
bow_tfidf_tit_txt_df_day = pd.DataFrame(bow_tfidf_tit_txt_day.transform(tit_txt_sentence_day).toarray(), columns = bow_tfidf_vocab_tit_txt_day)


## -------------------------------------- 2. Word2Vec 생성 ---------------
# Word2Vec : 단어를 벡터로 나타냄
w2v_tit_txt_day = Word2Vec(sentences=tit_txt_combination_date_day, vector_size=100, window=5, workers=4, sg=0)


## -------------------------------------- 3. FastText 생성 ---------------
fasttext_tit_txt_day = FastText(tit_txt_combination_date_day, vector_size=100, window=5,  workers=4, sg=1)


## -------------------------------------- 4. BERT 생성 ---------------
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
bert_tit_txt_day= bert_model.encode(tit_txt_sentence_day)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(original_news.title_text_bert.iloc[0])
indexed_text = tokenizer.convert_tokens_to_ids(tokenized_text)

id = 0
segment_ids = [0]*len(tokenized_text)
for i in range(len(tokenized_text)):
    if tokenized_text[i]!="[SEP]":
        segment_ids[i] = id
    else:
        segment_ids[i] = id
        id+=1