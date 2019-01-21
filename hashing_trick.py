import numpy as np
import pandas as pd
import random
from collections import defaultdict;
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

df = pd.read_csv("../input/combined_data_1.txt", header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1]);
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True].reset_index()
movie_np = []
movieID = 1

for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
  temp = np.full((1, i - j - 1), movieID);
  movie_np = np.append(movie_np, temp);
  movieID += 1;

last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movieID);
movie_np = np.append(movie_np, last_record);
df = df[pd.notnull(df['Rating'])];

df['Cust_Id'] = df['Cust_Id'].astype(int);
df['movieID'] = movie_np.astype(int);
f = ['count', 'mean']
movieSumm = df.groupby('movieID')['Rating'].agg(f);
custSumm = df.groupby('Cust_Id')['Rating'].agg(f);
movieSumm.index = movieSumm.index.map(int);
custSumm.index = custSumm.index.map(int);
dropMovieList = movieSumm[movieSumm['count'] < round(movieSumm['count'].quantile(1), 0)].index;
custList = custSumm[custSumm['count'] < round(custSumm['count'].quantile(1), 0)].index;
df = df[~df['movieID'].isin(dropMovieList)];
df = df[~df['Cust_Id'].isin(custList)];
df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='movieID');
df_p1 = pd.pivot_table(df, values='Rating', index='movieID', columns='Cust_Id');
allMovies = np.asarray(df_p1[6].index.values.tolist());
df_title = pd.read_csv('../input/movie_titles.csv', header=None,names=['movieID', 'Year', 'Name']);
df_title.set_index('movieID', inplace=True);

def get_seen_unSeenMovies_for(user):
  seenMovies = np.asarray(df_p1[user].dropna().index.values.tolist());
  unSeenMovies = np.asarray([i for i in allMovies if i not in seenMovies]);
  return seenMovies, unSeenMovies;

allUserID = np.asarray(df_p.index.values.tolist());
model_size = 5;
extractor = FeatureHasher(n_features=2 ** model_size);
model = SGDClassifier(loss="log", penalty="L2");
lSVC_model = LinearSVC(penalty='l2');


def generate_features(seenMovies, unSeenMovies):
  seenPairs = []
  unseenPairs = []
  for movieID1 in seenMovies:
    moviePairList = dict()
    for movieID2 in seenMovies:
      if movieID1 != movieID2:
        moviePairList["%s_%s_%s_%s" % (movieID1, df_title.loc[movieID1, 'Year'], movieID2, df_title.loc[movieID2, 'Year'])] = 1
    seenPairs.append(moviePairList)
  for movieID1 in random.sample(unSeenMovies, len(seenMovies)):
    moviePairList = dict()
    for movieID2 in seenMovies:
      moviePairList["%s_%s_%s_%s" % (movieID1, df_title.loc[movieID1, 'Year'], movieID2, df_title.loc[movieID2, 'Year'])] = 1
    unseenPairs.append(moviePairList)
  labels = np.hstack([np.ones(len(seenPairs)), np.zeros(len(unseenPairs))])
  return labels, (seenPairs, unseenPairs)

def train_logistic():
  for i in allUserID:
    seenMovies, unSeenMovies = get_seen_unSeenMovies_for(i);
    labels, (seenPairs, unseenPairs) = generate_features(seenMovies, unSeenMovies);
    seenFeatures = extractor.transform(seenPairs);
    unseenFeatures = extractor.transform(unseenPairs);
    features = sp.vstack([seenFeatures, unseenFeatures]);
    model.partial_fit(features, labels, classes=[0, 1]);

def test_logistic():
  probeFile = "";
  testing_dict = defaultdict(list);
  try:
    with open(probeFile,'r') as filereader:
      movieID = 0;
      for i,line in enumerate(filereader):
        line = line.replace("\n","").replace("\r","").split(":");
        if len(line) == 2:
          movieID = line[0];
        else:
          testing_dict[movieID].append(line[0]);
  except Exception as e:
    print("");
  #Testing for a given user
  seenMovies, unSeenMovies = get_seen_unSeenMovies_for(allUserID[400]);
  labels, (seenPairs, unseenPairs) = generate_features(seenMovies, unSeenMovies);
  seenFeatures = extractor.transform(seenPairs);
  unseenFeatures = extractor.transform(unseenPairs);
  features = sp.vstack([seenFeatures, unseenFeatures]);
  predLabels = model.predict(features)
  predicted_prob = model.predict_proba(features)
  auc = roc_auc_score(labels, predLabels)
  confusionMatrix = confusion_matrix(labels, predLabels)
  print("Model size", model_size, "auc", auc)
  print(confusionMatrix)

def linearSVMTrainer():
  allFeatures = [];
  allLabels = [];
  x = 0
  for i in allUserID:
    seenMovies, unSeenMovies = get_seen_unSeenMovies_for(allUserID[i]);
    labels, (seenPairs, unseenPairs) = generate_features(seenMovies, unSeenMovies);
    seenFeatures = extractor.transform(seenPairs);
    unseenFeatures = extractor.transform(unseenPairs);
    if x != 0:
      allFeatures = sp.vstack([allFeatures, seenFeatures, unseenFeatures]);
      allLabels = np.hstack([allLabels, labels]);
    else:
      allFeatures = sp.vstack([seenFeatures, unseenFeatures]);
      allLabels = labels;
      x = 1
  print(allFeatures.shape)
  print(allLabels.shape)
  lSVC_model.fit(allFeatures, allLabels);

def linearSVM_testing():
  probeFile = "";
  testing_dict = defaultdict(list);
  try:
    with open(probeFile,'r') as filereader:
      movieID = 0;
      for i,line in enumerate(filereader):
        line = line.replace("\n","").replace("\r","").split(":");
        if len(line) == 2:
          movieID = line[0];
        else:
          testing_dict[movieID].append(line[0]);
  except Exception as e:
    print("");
  #Testing for a given user
  seenMovies, unSeenMovies = get_seen_unSeenMovies_for(allUserID[400]);
  labels, (seenPairs, unseenPairs) = generate_features(seenMovies, unSeenMovies);
  seenFeatures = extractor.transform(seenPairs);
  unseenFeatures = extractor.transform(unseenPairs);
  features = sp.vstack([seenFeatures, unseenFeatures]);
  predLabels = lSVC_model.predict(features)
  auc = roc_auc_score(labels, predLabels)
  confusionMatrix = confusion_matrix(labels, predLabels)
  print("Model size", model_size, "auc", auc)
  print(confusionMatrix)