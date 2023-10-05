# DowJ 기준 Score 계산 : NBC, NYT -> NLTK, TextBlob

# library
import pandas as pd
import datetime
import yfinance as yf

def date_cut_day(dataframe_date_timestamp):
    dataframe_date_timestamp = datetime.datetime.strftime(dataframe_date_timestamp, "%Y-%m-%d")
    return dataframe_date_timestamp


def Score(dir, stock, start_date, end_date, col_name):
    # score data
    score = pd.read_csv(dir)
    score.columns = ["Date", col_name]

    # 2012-01-01 ~ 2022-05-01 Dow Jones 데이터 수집
    end_1_next = pd.to_datetime(end_date, errors="ignore") + datetime.timedelta(days=1)
    end_1_next = end_1_next.strftime('%Y-%m-%d')
    price_data = yf.download([stock], start=start_date, end=end_1_next)

    # 2011-12-30 ~ 2022-04-30 날짜 총 데이터
    start_2_previous = pd.to_datetime(start_date, errors="ignore") - datetime.timedelta(days=2)
    start_2_previous = start_2_previous.strftime('%Y%m%d')
    Date = pd.date_range(start=start_2_previous, end=pd.to_datetime(end_date).strftime('%Y%m%d'))
    Date = pd.DataFrame({"Date": Date.values})
    Date.Date = Date.Date.apply(lambda x: date_cut_day(x))

    # 개장일을 1로 설정
    opening = price_data.index.copy()
    opening = pd.DataFrame(opening)
    opening["Opening_Date"] = 1
    opening.Date = opening.Date.apply(lambda x: date_cut_day(x))

    # 총 날짜 데이터와 개장일 데이터를 결합 -> 개장일이 아닌 날짜는 0으로 설정
    set_news_data_date = pd.merge(Date, opening, how="left", left_on='Date', right_on="Date")
    set_news_data_date = set_news_data_date.where(pd.notnull(set_news_data_date), 0)
    set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: pd.to_datetime(x, errors="ignore"))

    # 각 날짜의 news는 어떤 날의 price를 예측하는 데 사용될지 지정
    standard = set_news_data_date.Date.iloc[len(set_news_data_date) - 1] + datetime.timedelta(days=1)
    set_news_data_date["Prediction_Date"] = 0
    for i in range(len(set_news_data_date) - 1, -1,
                   -1):  # set_news_data_date의 length-1 부터 0까지 -1만큼 생성하기           ** python은 인덱스가 (자리-1)임
        if i == (len(set_news_data_date) - 1):
            standard = set_news_data_date.Date[i]
            set_news_data_date.Prediction_Date[i] = standard + datetime.timedelta(days=1)
        elif (set_news_data_date.Opening_Date[i] == 1) & (set_news_data_date.Opening_Date[i + 1] == 1):
            standard = set_news_data_date.Date[i]
            set_news_data_date.Prediction_Date[i] = standard + datetime.timedelta(days=1)
        elif (i != 0):
            if ((set_news_data_date.Opening_Date[i] == 1) & (set_news_data_date.Opening_Date[i + 1] == 0) & (
                    set_news_data_date.Opening_Date[i - 1] == 0)):
                set_news_data_date.Prediction_Date[i] = standard + datetime.timedelta(days=1)
                standard = set_news_data_date.Date[i]
            else:
                set_news_data_date.Prediction_Date[i] = standard
        else:
            set_news_data_date.Prediction_Date[i] = standard
    set_news_data_date.Date = set_news_data_date.Date.apply(lambda x: date_cut_day(x))

    # news score data와 날짜 데이터를 결합 -> 예측 날짜에 맞춰 Score의 평균을 계산
    score = pd.merge(set_news_data_date, score, how="left", on="Date")
    score = score[['Date', col_name, "Prediction_Date"]]
    score = score.groupby('Prediction_Date').mean({col_name})
    end_1_previous = pd.to_datetime(end_date, errors="ignore") - datetime.timedelta(days=1)
    end_1_previous = end_1_previous.strftime('%Y-%m-%d')
    score = score[score.index <= end_1_previous]
    score = score.reset_index()

    score.to_csv(dir[0:len(dir) - 4] + "_DowJ.csv", index=False)

# NBC NLTK Score
dir = "data/NLTK_NBC_Score.csv"
stock = "^DJI"
start_date = "2012-01-01"
end_date = "2022-04-30"
col_name = "NBC_N_Score"
Score(dir, stock, start_date, end_date, col_name)

# NBC TextBlob Score
dir = "data/TextB_NBC_Score.csv"
stock = "^DJI"
start_date = "2012-01-01"
end_date = "2022-04-30"
col_name = "NBC_TB_Score"
Score(dir, stock, start_date, end_date, col_name)

# NBC BERT Score
dir = "data/BERT_NBC_Score.csv"
stock = "^DJI"
start_date = "2012-01-01"
end_date = "2022-04-30"
col_name = "NBC_B_Score"
Score(dir, stock, start_date, end_date, col_name)

# NYT NLTK Score
dir = "data/NLTK_NYT_Score.csv"
stock = "^DJI"
start_date = "2012-01-01"
end_date = "2022-04-30"
col_name = "NYT_N_Score"
Score(dir, stock, start_date, end_date, col_name)

# NYT TextBlob Score
dir = "data/TextB_NYT_Score.csv"
stock = "^DJI"
start_date = "2012-01-01"
end_date = "2022-04-30"
col_name = "NYT_TB_Score"
Score(dir, stock, start_date, end_date, col_name)

dir = "data/BERT_NYT_Score.csv"
stock = "^DJI"
start_date = "2012-01-01"
end_date = "2022-04-30"
col_name = "NYT_B_Score"
Score(dir, stock, start_date, end_date, col_name)