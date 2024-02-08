# -*- coding: utf-8 -*-
"""Movie Recommendation.ipynb


Original file is located at
    https://colab.research.google.com/drive/1aDXWbsPl4CA-3aD_TvF8qhL0lbCOf6du
"""

import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans

# Unzipping the dataset
with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Importing the dataset
movies_info = pd.read_csv('data/ml-latest-small/movies.csv')
ratings_info = pd.read_csv('data/ml-latest-small/ratings.csv')

# Movie ID to movie name mapping
movie_names_map = movies_info.set_index('movieId')['title'].to_dict()
num_users = len(ratings_info.userId.unique())
num_items = len(ratings_info.movieId.unique())

class RatingDataset(Dataset):
    def __init__(self):
        self.ratings = ratings_info.copy()
        users_list = ratings_info.userId.unique()
        movies_list = ratings_info.movieId.unique()
        self.user_id_map = {user: idx for idx, user in enumerate(users_list)}
        self.movie_id_map = {movie: idx for idx, movie in enumerate(movies_list)}
        self.idx_to_user_id = {idx: user for user, idx in self.user_id_map.items()}
        self.idx_to_movie_id = {idx: movie for movie, idx in self.movie_id_map.items()}
        self.ratings.movieId = ratings_info.movieId.apply(lambda x: self.movie_id_map[x])
        self.ratings.userId = ratings_info.userId.apply(lambda x: self.user_id_map[x])
        self.x_data = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y_data = self.ratings['rating'].values
        self.x_data, self.y_data = torch.tensor(self.x_data), torch.tensor(self.y_data)

    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index])

    def __len__(self):
        return len(self.ratings)

class MatrixFactorizationModel(torch.nn.Module):
    def __init__(self, num_users, num_items, num_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users, num_factors)
        self.item_factors = torch.nn.Embedding(num_items, num_factors)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        users, items = data[:,0], data[:,1]
        return (self.user_factors(users) * self.item_factors(items)).sum(1)

    def predict(self, user, item):
        user_embedding = self.user_factors(torch.tensor([user]))
        item_embedding = self.item_factors(torch.tensor([item]))
        return (user_embedding * item_embedding).sum()

# Hyperparameters
num_epochs = 50
use_cuda = torch.cuda.is_available()

# Model initialization
model = MatrixFactorizationModel(num_users, num_items, num_factors=8)

# Move model to GPU if available
if use_cuda:
    model = model.cuda()

# Loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# DataLoader for training data
training_set = RatingDataset()
train_loader = DataLoader(training_set, 128, shuffle=True)

# Training loop
for epoch in tqdm(range(num_epochs), leave=False):
    losses = []
    for x_data, y_data in train_loader:
        if use_cuda:
            x_data, y_data = x_data.cuda(), y_data.cuda()
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = loss_function(outputs.squeeze(), y_data.type(torch.float32))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Epoch {}: Loss: {:.4f}".format(epoch+1, sum(losses) / len(losses)))

# Extracting trained embeddings
user_embeddings = model.user_factors.weight.data.cpu().numpy()
item_embeddings = model.item_factors.weight.data.cpu().numpy()

# Clustering movie embeddings
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(item_embeddings)

# Displaying clusters
for cluster_id in range(num_clusters):
    print("Cluster #{}".format(cluster_id))
    movies_list = []
    for movie_index in np.where(kmeans.labels_ == cluster_id)[0]:
        movie_id = training_set.idx_to_movie_id[movie_index]
        rating_count = ratings_info.loc[ratings_info['movieId'] == movie_id].count()[0]
        movies_list.append((movie_names_map[movie_id], rating_count))
    top_movies = sorted(movies_list, key=lambda tup: tup[1], reverse=True)[:10]
    for movie in top_movies:
        print("\t{}".format(movie[0]))