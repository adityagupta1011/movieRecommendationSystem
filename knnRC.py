import random
import pandas as pd
import scipy.sparse as sp

from math import sqrt
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error


ratingsData = pd.read_csv('moviesSmall/ratings.csv', usecols=['userId', 'movieId', 'rating'])
# fill empty entries with 0
ratingsData['userId'] = ratingsData['userId'].fillna(0)
ratingsData['movieId'] = ratingsData['movieId'].fillna(0)
# fill empty entries with the mean of all ratings
ratingsData['rating'] = ratingsData['rating'].fillna(ratingsData['rating'].mean())

# get fraction of data to development
frac = 1.0
fracData = ratingsData.sample(frac=frac)


def createMatrix(getUsers, userMovieDict, numUsers,  numMovies):
    users = []
    movies = []
    interactions = []
    for userId, idx in enumerate(getUsers):
        users.extend([userId] * len(userMovieDict[idx]))
        movies.extend(userMovieDict[idx])
        interactions.extend([1.] * len(userMovieDict[idx]))

    matrix = sp.coo_matrix((interactions, (users, movies)), shape=(int(numUsers), int(numMovies))).tocsr()
    return matrix


# method to create training and testing datasets
def trainTestSplit(data, numUserTest, numUserTrain):
    userMovieDict = defaultdict(set)
    maxMovieId = 0
    for idx, row in data.iterrows():
        userMovieDict[row['userId']].add(row['movieId'])
        maxMovieId = max(maxMovieId, row['movieId'])

    userMovieDict = list(userMovieDict.values())
    usersNeeded = numUserTest + numUserTrain
    getUsers = random.sample(xrange(len(userMovieDict)), usersNeeded)

    trMatrix = createMatrix(getUsers[:numUserTrain], userMovieDict, numUserTrain, maxMovieId + 1)
    tsMatrix = createMatrix(getUsers[numUserTrain:], userMovieDict, numUserTest, maxMovieId + 1)
    return trMatrix, tsMatrix


# get rmse error value
def rmse(predictedRatings, actualRatings):
    return sqrt(mean_squared_error(predictedRatings, actualRatings))


# run knn and get the error
def KNN(metric, trData, tsData, K):
    scores = []
    knn = NearestNeighbors(metric=metric, algorithm="brute")
    knn.fit(trData)
    for k in K:
        print "Evaluating for k =", k, "neighbors"
        kNeighbors = knn.kneighbors(tsData, n_neighbors=k, return_distance=False)
        allScores = []
        allLabels = []
        for userId in xrange(tsData.shape[0]):
            userInfo = tsData[userId, :]

            _, movIndices = userInfo.nonzero()
            moviesSeen = set(movIndices)
            moviesNotSeen = set(xrange(tsData.shape[1])) - moviesSeen

            n_samples = min(len(moviesNotSeen), len(moviesSeen))
            rndMoviesSeen = random.sample(moviesSeen, n_samples)
            rndMoviesNotSeen = random.sample(moviesNotSeen, n_samples)

            indices = list(rndMoviesSeen)
            indices.extend(rndMoviesNotSeen)
            labels = [1] * n_samples
            labels.extend([0] * n_samples)

            neighbors = trData[kNeighbors[userId, :], :]
            predicted_scores = neighbors.mean(axis=0)
            for idx in indices:
                allScores.append(predicted_scores[0, idx])
            allLabels.extend(labels)
        print "RMSE:", (rmse(allLabels, allScores)), "\n"


train_data, test_data = trainTestSplit(fracData, 50, 550)
KNN("cosine", train_data, test_data, [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
