
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
predict = np.load(os.path.join(data_folder, 'channels_predict.npy'))
data_3d.shape, event.shape, predict.shape

# %%
for j in range(6):
    print(f'--------------- {j} ------------------')
    print(metrics.classification_report(y_true=event, y_pred=predict[:, j]))


# %%
