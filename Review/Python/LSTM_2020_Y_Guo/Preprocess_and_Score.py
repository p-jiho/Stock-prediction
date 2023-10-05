import nyt_collection_func as nyt_func
import datetime
import itertools
import pandas as pd
import numpy as np

import multiprocessing
from multiprocessing import Pool
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

year = list(map(str, list(range(2012,2023))))
month = list(map(str, list(range(1,13))))

year_month = []
for i in range(len(year)):
    for j in range(len(month)):
        year_month.append([year[i], month[j]])
year_month = year_month[0:124]

nyt_collection_result = list(map(lambda x:nyt_func.nyt_preprocess(x), year_month))
nyt_headline_date = list(map(lambda x: x[0], nyt_collection_result))
nyt_headline_token = list(map(lambda x: x[1], nyt_collection_result))
nyt_headline_date = list(itertools.chain.from_iterable(nyt_headline_date))
nyt_headline_token = list(itertools.chain.from_iterable(nyt_headline_token))

frequent_headline = pd.Series(list(itertools.chain(*nyt_headline_token))).value_counts()
headline_less_frequent_word = frequent_headline[frequent_headline<=10]
del frequent_headline

def del_less_frequent_headline(line):  ## 리스트의 한 line  ["a","b","c"]
    data = nyt_func.difference(line, list(set(line)& set(headline_less_frequent_word.index)))   ## 불용어 사전과 동일하게 각 line과 불용어의 교집합 사전을 만들어 해당되는 값 제거
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
    start = datetime.datetime.now()
    nyt_headline_token = main()
    print(datetime.datetime.now()-start)

for i in range(len(nyt_headline_token)-1):
    i += 1
    nyt_headline_token[0].extend(nyt_headline_token[i])
nyt_headline_token = nyt_headline_token[0]

nyt_headline_date = list(map(nyt_func.date_cut_day, nyt_headline_date))
nyt_headline_date_token = [[nyt_headline_date[i][1]] + nyt_headline_token[i] for i in range(len(nyt_headline_date))]
nyt_headline_date_token = sorted(nyt_headline_date_token, key=lambda date_plus_tit_txt: date_plus_tit_txt[0])
nyt_headline_token = list(map(nyt_func.from_combination_to_token, nyt_headline_date_token))
nyt_headline_sentence = list(map(nyt_func.word_to_sentence, nyt_headline_token))
nyt_date = [[nyt_headline_date[i][1]] for i in range(len(nyt_headline_date))]
nyt_headline_sentence = [nyt_date[i] + [nyt_headline_sentence[i]] for i in range(len(nyt_headline_sentence))]

analyzer = SentimentIntensityAnalyzer()
nyt_headline_sentence = [nyt_headline_sentence[i] + [analyzer.polarity_scores(nyt_headline_sentence[i][1])["compound"]] for i in range(len(nyt_headline_sentence))]

nyt_score = []
for i in range(len(nyt_headline_sentence)):
    if nyt_headline_sentence[i][1] != "":
        score = nyt_headline_sentence[i] + [TextBlob(nyt_headline_sentence[i][1]).sentences[0].sentiment.polarity]
    else : score = nyt_headline_sentence[i] + [0.0]
    nyt_score.append(score)

nyt_score = pd.DataFrame(nyt_score)
nyt_score.columns = ["Date","Headline", "N_Score", "B_Score"]
nyt_score = nyt_score[["Date", "N_Score", "B_Score"]]
nyt_score.to_csv("nyt_score.csv", index = False)