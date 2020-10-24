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

SERVING_URL = "http://18.180.157.88:8501/v1/models/ml-deepFM:predict"

data = pd.read_csv("./movielens_sample.txt")
sparse_features = ["movie_id", "user_id",
                   "gender", "age", "occupation", "zip"]
target = ['rating']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

X = data.to_json(orient='records', lines=True)
X = X.split("\n")
y = data['rating']

predict_data = []

i=1
for data in X:
    data = json.loads(data)
    data_new = {x: [y] for x, y in data.items()}
    request_body = {
        "signature_name": "serving_default",
        "instances": [data_new]
    }
    data_new.pop('timestamp')
    data_new.pop('rating')
    data_new.pop('title')
    data_new.pop('genres') # 这个实际上是 index 那一列
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

    print (f"{i}============")
    i=i+1
