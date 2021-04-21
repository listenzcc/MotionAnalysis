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

import threading

import plotly.express as px

# %%


def ravel(x):
    return np.array([e.ravel() for e in x])


def unravel(x):
    return np.array([e.reshape((20, 6)) for e in x])


def accumulate(x):
    assert(x.shape == (20, 6))
    y = np.zeros((20, 6))
    for i, _ in enumerate(x):
        y[i] = np.sum(x[:i+1], axis=0)
    return y


scaler = StandardScaler()

# %%
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

# %%
# Data_3d shape is (31625, 20, 6),
#                   samples x times x channels
# event shape is (31625,)
data_3d = np.load(os.path.join(data_folder, 'data3.npy'))
event = np.load(os.path.join(data_folder, 'event.npy'))
data_3d.shape, event.shape

# %%
# scaler = StandardScaler()
# data_3d = unravel(scaler.fit_transform(ravel(data_3d)))

# %%
data = ravel(np.array([accumulate(e) for e in data_3d]))
data.shape

# %%
label = event

clf = make_pipeline(StandardScaler(), SVC(gamma='scale'))
# clf = make_pipeline(SVC(gamma='scale'))
skf = StratifiedKFold(n_splits=10, shuffle=True)

predict = label * 0


class myThread(threading.Thread):
    def __init__(self, train_idx, test_idx):
        threading.Thread.__init__(self)
        self.X = data[train_idx]
        self.y = label[train_idx]
        self.X1 = data[test_idx]
        self.test_idx = test_idx

    def run(self):
        print('Thread starts')
        t = time.time()
        train_pred(self.X, self.y, self.X1, self.test_idx)
        print(f'Thread finished, costs {time.time() -t} seconds')


def train_pred(X, y, X1, test_idx):
    print(f'Fitting')
    clf.fit(X, y)

    print(f'Predicting')
    y2 = clf.predict(X1)

    predict[test_idx] = y2
    print(f'Done')


pool = []
for train_idx, test_idx in skf.split(data, label):
    thd = myThread(train_idx, test_idx)
    pool.append(thd)

for thd in pool:
    thd.start()

for thd in pool:
    thd.join()

print(metrics.classification_report(label, predict))

# %%
rdf = pd.DataFrame(columns=['label', 'predict'])
rdf['label'] = label
rdf['predict'] = predict
rdf.to_csv(os.path.join(
    data_folder, f'svm_accumulate_classification_results-{time.time()}.csv'))

# %%
print(metrics.classification_report(label, predict))
cm = metrics.confusion_matrix(label, predict, normalize='true')
px.imshow(cm)

# %%
