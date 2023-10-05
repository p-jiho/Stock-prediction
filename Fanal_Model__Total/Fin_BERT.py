# 이 코드는 제가 쓴 코드가 아닙니다. 코드 참조 https://www.kaggle.com/code/nlpquant/finbert-ext
# fin-BERT를 이용해 NYT, NBC 뉴스 데이터를 분석

import requests
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
# error : module 'numpy' has no attribute 'typeDict'
# 해결 : numpy == 1.21
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, TextClassificationPipeline, TFBertForSequenceClassification
from collections import Counter

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments,Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

f = open("Sentences_AllAgree.txt","r", encoding='UTF-8')
original_data = f.readlines()
f.close()
original_data = list(map(lambda x: x.strip().split("@"), original_data))
original_data = pd.DataFrame(original_data)
original_data.columns = ["Text","Label"]

label_encoder = LabelEncoder()
label_encoder.fit(original_data['Label'])

original_data['Label'] = np.asarray(label_encoder.transform(original_data['Label']), dtype=np.int32)
original_data.columns = ["text","label"]

dataset = Dataset.from_pandas(original_data[["text","label"]])
ds = dataset.train_test_split(test_size=0.2,seed=42)

tokz = AutoTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')

def tok_func(x): return tokz(x["text"],padding=True,truncation=True,max_length=128)
tok_ds = ds.map(tok_func, batched=True)
tok_ds = tok_ds.remove_columns('text')
def tok_func(x): return tokz(x["text"],padding=True,truncation=True,max_length=128)
tok_ds = ds.map(tok_func, batched=True)
tok_ds = tok_ds.remove_columns('text')

bs = 16
lr = 0.0001
epochs = 20

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall}

args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine',
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*4,
                         weight_decay=0.01, report_to='none', num_train_epochs=epochs, load_best_model_at_end = False,
                         logging_strategy='epoch'
)

model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-pretrain", num_labels=3)

trainer = Trainer(model, args, train_dataset=tok_ds['train'], eval_dataset=tok_ds['test'],
                  tokenizer=tokz,compute_metrics=compute_metrics)
trainer.train()

pred = trainer.predict(tok_ds['test'])
pred = np.argmax(pred.predictions, axis=-1)
val = ds['test'].to_pandas()
assert len(val)==len(pred)
val['pred'] = pred

print("------------------ Train accuracy : {} ------------------".format((val.label==val.pred).mean()))

f = open("data/NBC_News.txt","r", encoding = "UTF-8")
NBC_News = f.readlines()
f.close()


NBC_News = list(map(lambda x: x.split("%/%/%/%/%"), NBC_News))

NBC_News = pd.DataFrame(NBC_News)
sentence = NBC_News[1]
sentence = pd.DataFrame(sentence)
sentence.columns = ["text"]
sentence = Dataset.from_pandas(sentence)
sentence = sentence.map(tok_func, batched=True)
sentence = sentence.remove_columns('text')

preds = trainer.predict(sentence)
preds = np.argmax(preds.predictions, axis=-1)

NBC_News.columns = ["Date", "Text"]
NBC_News["Bert_Score"] = preds
NBC_News[["Date","Bert_Score"]].to_csv("BERT_NBC_Score.csv", header = True, index = None)
NBC_News.to_csv("data/BERT_NBC_Score.csv", header = True, index = None)

f = open("NYT_News.txt","r", encoding = "UTF-8")
NYT_News = f.readlines()
f.close()


NYT_News = list(map(lambda x: x.split("%/%/%/%/%"), NYT_News))
NYT_News = pd.DataFrame(NYT_News)
sentence = NYT_News[1]
sentence = pd.DataFrame(sentence)
sentence.columns = ["text"]
sentence = Dataset.from_pandas(sentence)
sentence = sentence.map(tok_func, batched=True)
sentence = sentence.remove_columns('text')

preds = trainer.predict(sentence)
preds = np.argmax(preds.predictions, axis=-1)

NYT_News.columns = ["Date", "Text"]
NYT_News["NYT_B_Score"] = preds
NYT_News[["Date","NYT_B_Score"]].to_csv("BERT_NYT_Score.csv", header = True, index = None)
NYT_News.to_csv("data/BERT_NYT_Score.csv", header = True, index = None)