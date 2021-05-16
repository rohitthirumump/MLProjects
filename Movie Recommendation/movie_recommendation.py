#Importing Libraries
import numpy as np
import pandas as pd

pip install fuzzywuzzy

#Importing Dataset
movies = pd.read_csv('movies.csv',usecols = ['movieId', 'title'])
ratings = pd.read_csv('ratings.csv', usecols = ['userId', 'movieId', 'rating'])

movies.head()

ratings.head()

#Creating matrix
rating_matrix = ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)
rating_matrix.head()

#Converting the matrix into csr_matrix
from scipy.sparse import csr_matrix
mat_movies = csr_matrix(rating_matrix.values)

##Implementing Knn model
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(metric = 'cosine',algorithm = 'brute',n_neighbors = 20)
model.fit(mat_movies)

##Implementing the recommender
from fuzzywuzzy import process
def recommenders(movie_name,data,n):
  index = process.extractOne(movie_name,movies['title'])[2]
  print('Movie Selected: ',movies['title'][index],'Index: ',index)
  print("Searching for recommendation.................")
  distance, indices = model.kneighbors(data[index],n_neighbors=n)
  for i in indices:
    print(movies['title'][i].where(i!= index))

movie_title = input("Enter movie title that you want the recommendation of: ")
recommenders(movie_title,mat_movies,10)
