# %%
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import threading

# %%
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
path = os.path.join(data_folder,
                    '处理后的数据(200ms)(6个人).csv')

# %%
df = pd.read_csv(path, header=None)
df

# %%
dataAll = df.values
label = dataAll[:, 0]
data = dataAll[:, 2:]

# %%
clf = make_pipeline(StandardScaler(), SVC(gamma='scale'))
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
    data_folder, f'raw_classification_results-{time.time()}.csv'))

# %%
print(metrics.classification_report(label, predict))
cm = metrics.confusion_matrix(label, predict, normalize='true')
px.imshow(cm)

# %%
