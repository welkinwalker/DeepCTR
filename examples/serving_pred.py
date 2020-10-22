"""
本文件主要用于向deepFM模型发起restful api请求， 获取ctr预估值
请求数据源来自于train_sample.txt，需要保证以下几点：
1. 逐条预测得到的ctr值与模型直接预测的值一模一样。
2. 一次预测的值与模型预测的一模一样。
3. 所有的请求预测得到的AUC与模型直接预测得到的一模一样。

pandas int64的type是无法被序列化的
"""

import os
import pandas as pd
import requests
import pickle
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

SERVING_URL = "http://localhost:8501/v1/models/deepFM:predict"

predict_data = "./merge.csv"
df = pd.read_csv(predict_data)

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

df[sparse_features] = df[sparse_features].fillna('-1', )
df[dense_features] = df[dense_features].fillna(0, )
target = ['label']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

X = df.to_json(orient='records', lines=True)
X = X.split("\n")
y = df['label']

predict_data = []

for data in X:
    data = json.loads(data)
    data_new = {x: [y] for x, y in data.items()}
    request_body = {
        "signature_name": "serving_default",
        "instances": [data_new]
    }
    data_new.pop('label')
    data_new.pop('pred')
    data_new.pop('Unnamed: 0') # 这个实际上是 index 那一列
    print(data_new)
    print(f"send request to tensorflow with data :{request_body}")
    headers = {"Content-type": "application/json"}
    data_post = json.dumps(request_body)
    print(data_post)
    response = requests.post(SERVING_URL, data=data_post, headers=headers)
    response_data = response.json()
    print(response_data)
    response.raise_for_status()
    predict_data.append(response_data['predictions'][0])

predict_data = np.array(predict_data)
auc_score = roc_auc_score(y, predict_data)
print(f"tensorflow restful api get auc score is {auc_score}")
