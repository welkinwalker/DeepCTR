import os
import pandas as pd
import requests
import pickle
import json, redis, time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

""" event sample
{
    "user_id":2077,
    "movie_id": [
        1221,
        3045,
        3374
  ]
}
"""
the_ip = 'localhost'
# the_ip = '172.31.26.126'

r = redis.Redis(host=the_ip, port=6379, db=0, password='')
SERVING_URL = f"http://{the_ip}:8501/v1/models/ml-deepFM:predict"


def lambda_handler(event, context):
    print(event)

    feat_dict = r.hgetall('ml-cate')

    predict_data = []
    data = {}
    age = int(event['age'])
    if age < 10:
        data['age'] = feat_dict['a1']
    elif age < 20:
        data['age'] = feat_dict['a18']
    elif age < 30:
        data['age'] = feat_dict['a25']
    elif age < 40:
        data['age'] = feat_dict['a35']
    elif age < 50:
        data['age'] = feat_dict['a45']
    elif age < 60:
        data['age'] = feat_dict['a56']
    else:
        return {
            'statusCode': 412,
            'body': f"age {event['age']} invalid"
        }

    occupation = int(event['occupation'])
    if 0 <= occupation <= 20:
        data['occupation'] = feat_dict['o' + str(occupation)]
    else:
        return {
            'statusCode': 413,
            'body': f"occupation {event['occupation']} invalid"
        }

    gender = event['gender']
    if gender == 'F' or gender == 'M':
        data['gender'] = feat_dict['g' + gender]
    else:
        return {
            'statusCode': 414,
            'body': f"gender {event['gender']} invalid"
        }

    data['age']=int(data['age'])
    data['gender']=int(data['gender'])
    data['occupation']=int(data['occupation'])

    m_list = ['m' + str(x) for x in event['movie_id']]
    m_strs = r.mget(m_list)

    m_idx = 0
    data_list = []
    for m_str in m_strs:
        if m_str is None:
            return {
                'statusCode': 402,
                'body': f"movie {event['movie_id'][m_idx]} invalid"
            }
        else:
            mdata = json.loads(m_str)
            data['movie_id'] = mdata['movie_id']

            data_new = {x: [y] for x, y in data.items()}
            data_list.append(data_new)
        m_idx = m_idx + 1

    request_body = {
        "signature_name": "serving_default",
        "instances": data_list
    }

    print(f"send request to tensorflow with data :{request_body}")
    headers = {"Content-type": "application/json"}
    data_post = json.dumps(request_body)
    print(data_post)

    time_start = time.time()
    response = requests.post(SERVING_URL, data=data_post, headers=headers)
    time_end = time.time()
    print(f"prediction time cost : {time_end - time_start}")
    response_data = response.json()
    print(response_data)

    response.raise_for_status()
    predict_data.append(response_data['predictions'][0])

    res_dict = {}
    m_idx = 0
    mid_list = event['movie_id']
    pred_list = response_data['predictions']
    while m_idx < len(mid_list):
        res_dict[mid_list[m_idx]] = pred_list[m_idx][0]
        m_idx = m_idx + 1

    return {
        'statusCode': 200,
        'body': res_dict
    }


event = {
    "user_id": 2077,
    "movie_id": [
        3045,
        1221,
        3092,
        3374
    ]
}

print(lambda_handler(event, None))
