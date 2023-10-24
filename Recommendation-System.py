from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
import os
import urllib.request
import zipfile

target_directory = 'D:/Intern 1/'

if not os.path.exists(target_directory):
    os.makedirs(target_directory)

url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

local_path = os.path.join(target_directory, 'ml-1m.zip')

if not os.path.exists(local_path):
    urllib.request.urlretrieve(url, local_path)

with zipfile.ZipFile(local_path, 'r') as zip_ref:
    zip_ref.extractall(target_directory)

reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file(os.path.join(target_directory, 'ml-1m/ratings.dat'), reader)

trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

user_id = input("Enter a User ID: ")
user_id = str(user_id)

# Load movie data
movies = pd.read_csv(os.path.join(target_directory, 'ml-1m/movies.dat'), sep='::', engine='python', encoding='ISO-8859-1', header=None, names=['item_id', 'movie_title', 'genres'])

# Get a list of all movie IDs the user has not rated
movies_rated_by_user = [str(movie_id) for (uid, movie_id, rating) in testset if uid == user_id]
all_movie_ids = [str(movie_id) for movie_id in movies['item_id'].unique() if str(movie_id) not in movies_rated_by_user]

testset_for_user = [(user_id, movie_id, 0) for movie_id in all_movie_ids]
predictions = model.test(testset_for_user)

top_n = [(iid, est) for (uid, iid, _, est, _) in predictions if uid == user_id]
top_n.sort(key=lambda x: x[1], reverse=True)

print(f"Top 10 Movie Recommendations for User {user_id}:")

for movie_id, rating in top_n[:10]:
    movie_title = movies[movies['item_id'] == int(movie_id)]['movie_title'].values[0]
    print(f'Movie: {movie_title}, Predicted Rating: {rating}')
