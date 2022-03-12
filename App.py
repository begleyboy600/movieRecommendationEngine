import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

ratings = pd.read_csv("ratings.csv")
print(ratings.head())

movies = pd.read_csv("movies.csv")
print(movies.head())

n_ratings = len(ratings)
n_movies = len(movies['movieId'].unique())
n_users = len(ratings['userId'].unique())

print(f"number of ratings: {n_ratings}")
print(f"number of unique movies: {n_movies}")
print(f"number of unique users: {n_users}")
print(f"average ratings per user: {round(n_ratings / n_users, 2)}")
print(f"average ratings per movie: {round(n_ratings / n_movies, 2)}")

user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
print(user_freq.head())

# find the lowest and highest rated movies:
mean_rating = ratings.groupby('movieId')[['rating']].mean()

# lowest rated movies
lowest_rated = mean_rating['rating'].idxmin()
movies.loc[movies['movieId'] == lowest_rated]

# highest rated movie
highest_rated = mean_rating['rating'].idxmin()
movies.loc[movies['movieId'] == highest_rated]

# show number of people who rated movies rated movie highest
ratings[ratings['movieId'] == highest_rated]

# show number of people who rated movies rated movie lowest
ratings[ratings['movieId'] == lowest_rated]

# the above movies has very low dataset, so therefore we will use bayesian average
movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()


# now, we create user-item matrix using scipy csr matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    # map Ids to indices
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    x = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return x, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


x, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# find similar movies using KNN

def find_similar_movies(movie_id, x, k, metric='cosine', show_distance=False):
    neighbour_ids = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = x[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(x)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids


movies_titles = dict(zip(movies['movieId'], movies['title']))
movie_id = 3

similar_ids = find_similar_movies(movie_id, x, k=10)
movies_title = movies_titles[movie_id]

print(f"since you watched {movies_title}")
for i in similar_ids:
    print(movies_titles[i])












