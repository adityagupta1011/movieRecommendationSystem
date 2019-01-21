#Movie_recommendation systems using svd
import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, evaluate
import os

data_frame1 = pd.read_csv('input/combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
data_frame1['Rating'] = data_frame1['Rating'].astype(float)
#Similarly we can upload the combined_data_2 , 3 and 4 files into data frames
#Due to the computing limitations of our laptops we have decided to use 1/4th of data set

#print('The shape of the data frame is {}'.format(data_frame1.shape))
print('Displaying data set examples')
print(data_frame1.iloc[::5000000, :])


df = data_frame1
#df = df.append(data_frame2)
#df = df.append(data_frame3)
#df = df.append(data_frame3)

df.index = np.arange(0,len(df))
#trying to print the data set examples
#print(df.iloc[::5000000, :])

p = df.groupby('Rating')['Rating'].agg(['count'])
movie_count = df.isnull().sum()[1]
cust_count = df['Cust_Id'].nunique() - movie_count
rating_count = df['Cust_Id'].count() - movie_count
nan_data_frame = pd.DataFrame(pd.isnull(df.Rating))
nan_data_frame = nan_data_frame[nan_data_frame['Rating'] == True]
nan_data_frame = nan_data_frame.reset_index()
movie_np = []
movie_id = 1

for i,j in zip(nan_data_frame['index'][1:],nan_data_frame['index'][:-1]):
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1


last_record = np.full((1,len(df) - nan_data_frame.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)


df = df[pd.notnull(df['Rating'])]
df['Movie_Id'] = movie_np.astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)


f = ['count','mean']

#getting rid of the useless movies and users
#This was done using by ignoring the users and movies with minimum review counts
data_frame_summary = df.groupby('Movie_Id')['Rating'].agg(f)
data_frame_summary.index = data_frame_summary.index.map(int)
bcmark_movies = round(data_frame_summary['count'].quantile(0.8),0)
filter_movies = data_frame_summary[data_frame_summary['count'] < bcmark_movies].index

filter_users = df.groupby('Cust_Id')['Rating'].agg(f)
filter_users.index = filter_users.index.map(int)
cust_benchmark = round(filter_users['count'].quantile(0.8),0)
drop_cust_list = filter_users[filter_users['count'] < cust_benchmark].index

#print('Customer that have given minimum numbers of review and are useless: {}'.format(cust_benchmark))

df = df[~df['Movie_Id'].isin(filter_movies)]
df = df[~df['Cust_Id'].isin(drop_cust_list)]

#print(df.iloc[::5000000, :])

df = df[~df['Movie_Id'].isin(filter_movies)]
df = df[~df['Cust_Id'].isin(drop_cust_list)]

df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')
df_title = pd.read_csv('input/movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)

reader = Reader()

#We are using only Top 100000 here because of very high running time
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)
data.split(n_folds=5)

svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
#TO DO -> Test against all 4 files if it is taking time consider one test file and one training file and run it twice

#Pass on the user id and ratings to this function to get the recommendations, however usually the rating is kept as 5 instead of passing it as a paramter 
def recommend_movies(custid, rat):
    df_user = df[(df['Cust_Id'] == custid) & (df['Rating'] == rat)]
    df_user = df_user.set_index('Movie_Id')
    df_user = df_user.join(df_title)['Name']
    user_copy = df_title.copy()
    user_copy = user_copy.reset_index()
    user_copy = user_copy[~user_copy['Movie_Id'].isin(drop_movie_list)]
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)
    trainset = data.build_full_trainset()
    svd.train(trainset)
    user_copy['Estimate_Score'] = user_copy['Movie_Id'].apply(lambda x: svd.predict(785314, x).est)
    user_copy = user_copy.drop('Movie_Id', axis = 1)
    user_copy = user_copy.sort_values('Estimate_Score', ascending=False)
    print(user_copy.head(10))











