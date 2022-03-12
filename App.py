import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
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



