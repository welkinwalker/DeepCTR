import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import redis

data = pd.read_csv("./movielens_sample.txt")
# sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
sparse_features = ["gender", "age", "occupation", "zip"]
target = ['rating']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
tmpu = data['user_id'].copy()
data['user_id'] = lbe.fit_transform(data['user_id'])
data['ouid'] = tmpu

tmpm = data['movie_id'].copy()
data['movie_id'] = lbe.fit_transform(data['movie_id'])
data['omid'] = tmpm

dict = {}
for index, row in data.iterrows():
    # if row.user_id in dict:
    #     print(row.to_json())
    #     print(dict[row.user_id])
    #     print("=================")
    # dict[row.user_id] = row.to_json()

    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set(f"u{row.ouid}", row.to_json())
    r.set(f"m{row.omid}", row.to_json())
