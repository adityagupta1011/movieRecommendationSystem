import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from sklearn import model_selection as ms
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

ratingsData = pd.read_csv('moviesSmall/ratings.csv', usecols=['userId', 'movieId', 'rating'])
# fill empty entries with 0
ratingsData['userId'] = ratingsData['userId'].fillna(0)
ratingsData['movieId'] = ratingsData['movieId'].fillna(0)
# fill empty entries with the mean of all ratings
ratingsData['rating'] = ratingsData['rating'].fillna(ratingsData['rating'].mean())

# get fraction of data to development
fraction = 1.0
fracData = ratingsData.sample(frac=fraction)

trData, tsData = ms.train_test_split(fracData, test_size=0.2)

# convert pandas dataframe to numpy matrix
trMatrix = trData.as_matrix(columns=['userId', 'movieId', 'rating'])
tsMatrix = tsData.as_matrix(columns=['userId', 'movieId', 'rating'])

# generate user-user similarity matrix
userPC = 1 - pairwise_distances(trData, metric='correlation')
userPC[np.isnan(userPC)] = 0

# generate item-item similarity matrix
itemPC = 1 - pairwise_distances(trData.T, metric='correlation')
itemPC[np.isnan(itemPC)] = 0


# predict the user's rating
def predict(trainingData, PCMatrix, type='u'):
    if type == 'i':
        predictedRatings = trainingData.dot(PCMatrix) / np.array([np.abs(PCMatrix).sum(axis=1)])
    elif type == 'u':
        meanUserRating = trainingData.mean(axis=1)
        diff = (trainingData - meanUserRating[:, np.newaxis])
        predictedRatings = meanUserRating[:, np.newaxis] + PCMatrix.dot(diff) / np.array([np.abs(PCMatrix).sum(axis=1)]).T
    return predictedRatings

userBasedPrediction = predict(trMatrix, userPC)
itemBasedPrediction = predict(trMatrix, itemPC, type='i')


# calculate the root mean squared error
def rmse(predictedRating, actualRating):
    predictedRating = predictedRating[actualRating.nonzero()].flatten()
    actualRating = actualRating[actualRating.nonzero()].flatten()
    return sqrt(mean_squared_error(predictedRating, actualRating))

print('Using ' + str(fraction) + ' fraction of total data. Results: ')
print('RMSE for user based Collaborative Filtering: ' + str(rmse(userBasedPrediction, trMatrix)))
print('RMSE for item based Collaborative Filtering: ' + str(rmse(itemBasedPrediction, trMatrix)))
