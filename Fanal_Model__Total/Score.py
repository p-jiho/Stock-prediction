# NBC, NYT의 NLTK, TextBlob Score

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd

# NBC NLTK 감성분석
f = open("data/NBC_News.txt", "r", encoding="UTF-8")
NBC_News = f.readlines()
f.close()

NBC_News = list(map(lambda x: x.split("%/%/%/%/%"), NBC_News))
for i in range(len(NBC_News)):
    NBC_News[i][1] = NBC_News[i][1].strip()

analyzer = SentimentIntensityAnalyzer()
NBC_News = [NBC_News[i] + [analyzer.polarity_scores(NBC_News[i][1])["compound"]] for i in range(len(NBC_News))]
NBC_News = pd.DataFrame(NBC_News)
NBC_News.columns = ["Date", "Title_Text", "NBC_N_Score"]
NBC_News = NBC_News[["Date", "NBC_N_Score"]]
NBC_News.to_csv("data/NLTK_NBC_Score.csv", index=False)

# NBC TextBlob 감성분석
f = open("data/NBC_News.txt", "r", encoding="UTF-8")
NBC_News = f.readlines()
f.close()

NBC_News = list(map(lambda x: x.split("%/%/%/%/%"), NBC_News))
for i in range(len(NBC_News)):
    NBC_News[i][1] = NBC_News[i][1].strip()

TextB_score = []
for i in range(len(NBC_News)):
    if NBC_News[i][1] != "":
        score = NBC_News[i] + [TextBlob(NBC_News[i][1]).sentences[0].sentiment.polarity]
    else:
        score = NBC_News[i] + [0.0]
    TextB_score.append(score)

TextB_score = pd.DataFrame(TextB_score)
TextB_score.columns = ["Date", "Text_Title", "NBC_TB_Score"]
TextB_score = TextB_score[["Date", "NBC_TB_Score"]]
TextB_score.to_csv("data/TextB_NBC_Score.csv", index=False)

# NYT NLTK 감성분석
f = open("data/NYT_News.txt","r", encoding = "UTF-8")
NYT_News = f.readlines()
f.close()

NYT_News = list(map(lambda x: x.split("%/%/%/%/%"), NYT_News))
for i in range(len(NYT_News)):
    NYT_News[i][1] = NYT_News[i][1].strip()

analyzer = SentimentIntensityAnalyzer()
NYT_News = [NYT_News[i] + [analyzer.polarity_scores(NYT_News[i][1])["compound"]] for i in range(len(NYT_News))]
NYT_News = pd.DataFrame(NYT_News)
NYT_News.columns = ["Date","Title_Text", "NYT_N_Score"]
NYT_News = NYT_News[["Date","NYT_N_Score"]]
NYT_News.to_csv("data/NLTK_NYT_Score.csv", index = False)

# NYT TextBlob 감성분석
f = open("data/NYT_News.txt", "r", encoding="UTF-8")
NYT_News = f.readlines()
f.close()

NYT_News = list(map(lambda x: x.split("%/%/%/%/%"), NYT_News))
for i in range(len(NYT_News)):
    NYT_News[i][1] = NYT_News[i][1].strip()

TextB_score = []
for i in range(len(NYT_News)):
    if NYT_News[i][1] != "":
        score = NYT_News[i] + [TextBlob(NYT_News[i][1]).sentences[0].sentiment.polarity]
    else:
        score = NYT_News[i] + [0.0]
    TextB_score.append(score)

TextB_score = pd.DataFrame(TextB_score)
TextB_score.columns = ["Date", "Text_Title", "NYT_TB_Score"]
TextB_score = TextB_score[["Date", "NYT_TB_Score"]]
TextB_score.to_csv("data/TextB_NYT_Score.csv", index=False)
