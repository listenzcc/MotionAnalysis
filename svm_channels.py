# %%
import os
import time
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tqdm.auto import tqdm

import threading

import plotly.express as px

# %%


def ravel(x):
    return np.array([e.ravel() for e in x])


def unravel(x):
    return np.array([e.reshape((20, 6)) for e in x])


scaler = StandardScaler()

# %%
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

# %%
# Data_3d shape is (31625, 20, 6),
#                   samples x times x channels
# event shape is (31625,)
data_3d = np.load(os.path.join(data_folder, 'data3.npy'))
event = np.load(os.path.join(data_folder, 'event.npy'))

scaler = StandardScaler()
data_3d = unravel(scaler.fit_transform(ravel(data_3d)))

data_3d.shape, event.shape

# %%
label = event

# clf = make_pipeline(StandardScaler(), SVC(gamma='scale'))
clf = make_pipeline(SVC(gamma='scale'))
skf = StratifiedKFold(n_splits=10, shuffle=True)

s = data_3d.shape
predict = np.zeros((s[0], s[2]))


class myThread(threading.Thread):
    def __init__(self, train_idx, test_idx):
        threading.Thread.__init__(self)
        self.X = data_3d[train_idx]
        self.y = label[train_idx]
        self.X1 = data_3d[test_idx]
        self.test_idx = test_idx

    def run(self):
        print('Thread starts')
        t = time.time()
        train_pred(self.X, self.y, self.X1, self.test_idx)
        print(f'Thread finished, costs {time.time() -t} seconds')


def train_pred(X, y, X1, test_idx):
    for i in tqdm(range(X.shape[2])):
        print(f'Fitting')
        print(X[:, :, i].shape)
        clf.fit(X[:, :, i], y)

        print(f'Predicting')
        y2 = clf.predict(X1[:, :, i])

        predict[test_idx, i] = y2
        print(f'Done')


pool = []
for train_idx, test_idx in skf.split(data_3d, label):
    thd = myThread(train_idx, test_idx)
    pool.append(thd)

for thd in pool:
    thd.start()

for thd in pool:
    thd.join()

# print(metrics.classification_report(label, predict))

# %%

np.save(os.path.join(data_folder, 'channels_predict.npy'), predict)

# %%
