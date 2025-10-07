import os
import urllib.request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# STEP 1: DOWNLOAD MOVIELENS DATASET
# ----------------------------------------------------------------------
if not os.path.exists("u.data"):
    print("Downloading MovieLens 100k dataset... ⏬")
    urllib.request.urlretrieve(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
        "u.data"
    )
if not os.path.exists("u.item"):
    urllib.request.urlretrieve(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.item",
        "u.item"
    )
print("Dataset ready ✅")

# ----------------------------------------------------------------------
# STEP 2: LOAD DATA
# ----------------------------------------------------------------------
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=column_names)
n_users = ratings.user_id.nunique()
n_movies = ratings.movie_id.nunique()

# Create user-movie matrix
data_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
data_matrix = torch.FloatTensor(data_matrix.values)

# ----------------------------------------------------------------------
# STEP 3: DEFINE RBM MODEL
# ----------------------------------------------------------------------
class RBM:
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) * 0.1
        self.a = torch.zeros(1, nh)  # hidden bias
        self.b = torch.zeros(1, nv)  # visible bias
        self.nh = nh
        self.nv = nv

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk, lr=0.01):
        self.W += lr * (torch.mm(ph0.t(), v0) - torch.mm(phk.t(), vk))
        self.b += lr * torch.sum(v0 - vk, 0)
        self.a += lr * torch.sum(ph0 - phk, 0)

# ----------------------------------------------------------------------
# STEP 4: TRAIN RBM
# ----------------------------------------------------------------------
nv = n_movies
nh = 100
batch_size = 100
rbm = RBM(nv, nh)
epochs = 10

for epoch in range(epochs):
    loss_ = 0
    for i in range(0, n_users - batch_size, batch_size):
        v0 = data_matrix[i:i+batch_size]
        vk = v0.clone()
        ph0, _ = rbm.sample_h(v0)
        for k in range(5):  # Gibbs sampling steps
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 == 0] = v0[v0 == 0]  # Keep original ratings fixed
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        loss_ += torch.sum((v0 - vk) ** 2)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_:.2f}')

# ----------------------------------------------------------------------
# STEP 5: MAKE RECOMMENDATIONS FOR USER 1
# ----------------------------------------------------------------------
user_id = 0
v = data_matrix[user_id:user_id+1]
ph, _ = rbm.sample_h(v)
predicted_ratings, _ = rbm.sample_v(ph)

already_rated = v[0].nonzero().flatten()
recommendations = [(i, predicted_ratings[0, i].item()) for i in range(n_movies) if i not in already_rated]
recommendations.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 movie recommendations for User 1:")
for movie_id, rating in recommendations[:5]:
    print(f"Movie ID: {movie_id + 1}, Predicted Rating: {rating:.2f}")

print("\n✅ RBM Movie Recommender executed successfully in Codespace!")