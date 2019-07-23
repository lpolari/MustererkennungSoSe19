#%%
#Lars Parmakerli
#Jeremias Gayer


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


#%%

k = 3

#%%

def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=" ")
  X = df.iloc[:, 1:257].values
  y = df.iloc[:, 0].values.astype(int)
  return X, y

#%%

X, y = load_from_file("data/zip.train")
X_test, y_test = load_from_file("data/zip.test")

print(X.dtype)
print(y.dtype)

#%%

def euclidean_distance(a, b):
  return np.linalg.norm(a - b)

#%%

def knn(a, X, y, k):
  norms = np.apply_along_axis(lambda x: euclidean_distance(a, x), 1, X)
  nn_indices = np.argpartition(norms, k)[:k]
  nn_labels = y.take(nn_indices)
  nn_label_count = np.bincount(nn_labels)
  return( np.argmax(nn_label_count))

#%%

y_test_pred = np.apply_along_axis(lambda x: knn(x, X, y, k), 1, X_test)


#%%

confusion = np.zeros([10,10])

for i in range(y_test.shape[0]):
  confusion[y_test[i], y_test_pred[i]] = confusion[y_test[i], y_test_pred[i]] + 1
  print(confusion[y_test[i], y_test_pred[i]])

#%%

print(confusion)

