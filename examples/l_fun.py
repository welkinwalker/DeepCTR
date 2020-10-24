import os
import pandas as pd
import requests
import pickle
import json, redis
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
the_ip = '172.31.26.126'

r = redis.Redis(host=the_ip, port=6379, db=0, password='mvp')
SERVING_URL = f"http://{the_ip}:8501/v1/models/ml-deepFM:predict"


def lambda_handler(event, context):
    print(event)

    # import requests as req
    # resp = req.get(" http://18.180.157.88:8501/v1/models/deepFM/metadata")
    # print(resp.text)

    predict_data = []
    # data_criteo = '{"Unnamed: 0":1,"label":0,"I1":0.0,"I2":0.0029980013,"I3":0.0132669983,"I4":0.1086956522,"I5":0.6522125918,"I6":0.0,"I7":0.0,"I8":0.0714285714,"I9":0.005800464,"I10":0.0,"I11":0.0,"I12":0.0,"I13":0.1086956522,"C1":0,"C2":17,"C3":35,"C4":5,"C5":1,"C6":4,"C7":2,"C8":1,"C9":1,"C10":5,"C11":31,"C12":35,"C13":7,"C14":6,"C15":5,"C16":29,"C17":1,"C18":24,"C19":0,"C20":0,"C21":19,"C22":0,"C23":0,"C24":8,"C25":0,"C26":0,"pred":"[0.4257229268550873]"}'

    u_str = r.get(f"u{event['user_id']}")
    m_list = ['m' + str(x) for x in event['movie_id']]
    m_strs = r.mget(m_list)

    data_new = {}
    for m_str in m_strs:
        udata = json.loads(u_str)
        mdata = json.loads(m_str)
        data = {}
        data['user_id'] = udata['user_id']
        data['gender'] = udata['gender']
        data['age'] = udata['age']
        data['occupation'] = udata['occupation']
        data['zip'] = udata['zip']
        data['movie_id'] = mdata['movie_id']

        data_new={}
        if len(data_new) == 0:
            data_new = {x: [y] for x, y in data.items()}
        else:
            for x, y in data.items():
                ll = data_new[x]
                ll.append(y)
                data_new[x] = ll

    request_body = {
        "signature_name": "serving_default",
        "instances": [data_new]
    }

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

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


event = {
    "user_id": 2077,
    "movie_id": [
        1221,
        3045,
        3374
    ]
}

lambda_handler(event, None)
