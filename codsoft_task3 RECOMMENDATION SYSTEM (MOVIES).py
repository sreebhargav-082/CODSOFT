# -*- coding: utf-8 -*-
"""Movie Recommendation.ipynb

Automatically generated by Colaboratory.

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
with zipfile.ZipFile('ml-latest-small.zip', 'r') as a:
    a.extractall('data')

# Importing the dataset
b = pd.read_csv('data/ml-latest-small/movies.csv')
c = pd.read_csv('data/ml-latest-small/ratings.csv')

# Movie ID to movie name mapping
d = b.set_index('movieId')['title'].to_dict()
e = len(c.userId.unique())
f = len(c.movieId.unique())

class g(Dataset):
    def __init__(self):
        self.h = c.copy()
        i = c.userId.unique()
        j = c.movieId.unique()
        self.k = {user: idx for idx, user in enumerate(i)}
        self.l = {movie: idx for idx, movie in enumerate(j)}
        self.m = {idx: user for user, idx in self.k.items()}
        self.n = {idx: movie for movie, idx in self.l.items()}
        self.h.movieId = c.movieId.apply(lambda x: self.l[x])
        self.h.userId = c.userId.apply(lambda x: self.k[x])
        self.o = self.h.drop(['rating', 'timestamp'], axis=1).values
        self.p = self.h['rating'].values
        self.o, self.p = torch.tensor(self.o), torch.tensor(self.p)

    def __getitem__(self, index):
        return (self.o[index], self.p[index])

    def __len__(self):
        return len(self.h)

class q(torch.nn.Module):
    def __init__(self, num_users, num_items, num_factors=20):
        super().__init__()
        self.r = torch.nn.Embedding(num_users, num_factors)
        self.s = torch.nn.Embedding(num_items, num_factors)
        self.r.weight.data.uniform_(0, 0.05)
        self.s.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        t, u = data[:,0], data[:,1]
        return (self.r(t) * self.s(u)).sum(1)

    def predict(self, user, item):
        user_embedding = self.r(torch.tensor([user]))
        item_embedding = self.s(torch.tensor([item]))
        return (user_embedding * item_embedding).sum()

# Hyperparameters
v = 50
w = torch.cuda.is_available()

# Model initialization
x = q(e, f, num_factors=8)

# Move model to GPU if available
if w:
    x = x.cuda()

# Loss function and optimizer
y = torch.nn.MSELoss()
z = torch.optim.Adam(x.parameters(), lr=1e-3)

# DataLoader for training data
aa = g()
ab = DataLoader(aa, 128, shuffle=True)

# Training loop
for ac in tqdm(range(v), leave=False):
    ad = []
    for ae, af in ab:
        if w:
            ae, af = ae.cuda(), af.cuda()
        z.zero_grad()
        ag = x(ae)
        ah = y(ag.squeeze(), af.type(torch.float32))
        ad.append(ah.item())
        ah.backward()
        z.step()
    print("Epoch {}: Loss: {:.4f}".format(ac+1, sum(ad) / len(ad)))

# Extracting trained embeddings
ai = x.r.weight.data.cpu().numpy()
aj = x.s.weight.data.cpu().numpy()

# Clustering movie embeddings
num_clusters = 5
ak = KMeans(n_clusters=num_clusters, random_state=0).fit(aj)

# Displaying clusters
for al in range(num_clusters):
    print("Cluster #{}".format(al))
    am = []
    for an in np.where(ak.labels_ == al)[0]:
        ao = aa.n[an]
        ap = c.loc[c['movieId'] == ao].count()[0]
        am.append((d[ao], ap))
    top_movies = sorted(am, key=lambda tup: tup[1], reverse=True)[:10]
    for aq in top_movies:
        print("\t{}".format(aq[0]))