# https://velog.io/@jaehyeong/Fine-tuning-Bert-using-Transformers-and-TensorFlow
# 이 코드는 위 링크의 코드를 상당히 많이 참고했습니다.
# fin-BERT와 위 링크의 코드를 결합해 NBC, NYT의 감성점수를 내는 코드입니다.
# 문제점 : NBC의 경우 길이가 너무 길어 분석 X -> title만 뽑아서 분석해볼 예정

#https://huggingface.co/yiyanghkust/finbert-pretrain
import requests
import json
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, TextClassificationPipeline, TFBertForSequenceClassification
from collections import Counter

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import requests
import json
import pandas as pd
import numpy as np

f = open("data/Sentences_AllAgree.txt","r", encoding='UTF-8')
original_data = f.readlines()
f.close()
original_data = list(map(lambda x: x.strip().split("@"), original_data))
original_data = pd.DataFrame(original_data)
original_data.columns = ["Text","Label"]

original_data[["Label_n"]] = 0

for i in range(len(original_data)):
    if original_data["Label"][i] == "neutral":
        original_data["Label_n"][i] = 1
    elif original_data["Label"][i] == "negative":
        original_data["Label_n"][i] = 0
    else : original_data["Label_n"][i] = 2

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-pretrain", do_lower_case=True)

Text = original_data["Text"].to_list()
Lable = original_data["Label_n"].to_list()

Train_Text, Test_Text, Train_Label, Test_Label = train_test_split(Text, Lable, test_size=0.3, random_state = 42)
Train_Text, Eval_Text, Train_Label, Eval_Label = train_test_split(Train_Text, Train_Label, test_size=0.1, random_state = 42)

Train_encoding = tokenizer(Train_Text, truncation=True, padding=True)
Test_encoding = tokenizer(Test_Text, truncation=True, padding=True)
Eval_encoding = tokenizer(Eval_Text, truncation=True, padding=True)

# trainset-set
Train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Train_encoding),
    Train_Label
))

# validation-set
Test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Test_encoding),
    Test_Label
))

Eval_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Eval_encoding),
    Eval_Label
))

model = TFBertForSequenceClassification.from_pretrained("yiyanghkust/finbert-pretrain", num_labels=3, from_pt=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

callback_earlystop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001, # the threshold that triggers the termination (acc should at least improve 0.001)
    patience=3)
model.fit(Train_dataset.shuffle(100).batch(16), epochs=5, batch_size=16,
          validation_data=Eval_dataset.shuffle(100).batch(16),
          callbacks = [callback_earlystop])

text_classifier = TextClassificationPipeline(
    tokenizer=tokenizer,
    model=model,
    framework='tf',
    return_all_scores=True
)

predicted_label_list = []
i = 0
for text in Test_Text:
    preds_list = text_classifier(text)[0]
    sorted_preds_list = sorted(preds_list, key=lambda x: x['score'], reverse=True)
    predicted_label_list.append(sorted_preds_list[0]["label"])

predicted_label_list = list(map(lambda x: int(x[len(x) - 1:len(x)]), predicted_label_list))
sum(np.equal(predicted_label_list, Test_Label)) / len(predicted_label_list)

f = open("data/NBC_News.txt", "r", encoding="UTF-8")
NBC_News = f.readlines()
f.close()

NBC_News = list(map(lambda x: x.split("%/%/%/%/%"), NBC_News))

NBC_News = pd.DataFrame(NBC_News)
sentence = NBC_News[1].to_list()
sentence = list(map(lambda x: x.strip(), sentence))
sentence_pred = []
i = 0
for text in sentence:
    i += 1
    if i % 1000 == 0:
        print(i)
    preds_list = text_classifier(text)[0]
    sorted_preds_list = sorted(preds_list, key=lambda x: x['score'], reverse=True)
    sentence_pred.append(sorted_preds_list[0]["label"])

sentence_pred = list(map(lambda x: int(x[len(x) - 1:len(x)]), sentence_pred))
print("Result Accuracy : {}".format(sentence_pred))
