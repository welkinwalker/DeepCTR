import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import redis

data = pd.read_csv("./ml-1m/merge.dat")
# data = pd.read_csv("./movielens_sample.txt")
# sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
sparse_features = ["gender", "age", "occupation", "zip"]
target = ['rating']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat+'t'] = lbe.fit_transform(data[feat])


tmpm = data['movie_id'].copy()
data['movie_id'] = lbe.fit_transform(data['movie_id'])
data['omid'] = tmpm

dict = {}
r = redis.Redis(host='localhost', port=6379, db=0)
k_set = set([])
for index, row in data.iterrows():

    dict[f'g{row.gender}']=row.gendert
    dict[f'a{row.age}']=row.aget
    dict[f'o{row.occupation}']=row.occupationt

    if f"m{row.omid}" not in k_set:
        r.set(f"m{row.omid}", row.to_json())
        k_set.add(f"m{row.omid}")

r.hmset('ml-cate',dict)