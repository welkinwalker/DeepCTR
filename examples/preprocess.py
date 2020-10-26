import pandas as pd

# df_rating = pd.read_csv('ml-1m/ratings.dat', sep="::",engine='python')
# df_sample = df_rating.sample(frac=0.01).sort_values(by=['user_id','timestamp'])
# df_sample.to_csv('ml-1m/sample.dat',index=False)


# df_sample = pd.read_csv('ml-1m/sample.dat')
df_rating = pd.read_csv('ml-1m/ratings.dat', sep="::",engine='python')
df_user = pd.read_csv('ml-1m/users.dat', sep="::", engine='python')
df_movie = pd.read_csv('ml-1m/movies.dat', sep="::", engine='python')

# df_res = df_rating.merge(df_movie).merge(df_user).sort_values(by=['user_id','timestamp'])
df_res = df_rating.merge(df_movie).merge(df_user)
df_res.to_csv('ml-1m/merge.dat',index=False)
