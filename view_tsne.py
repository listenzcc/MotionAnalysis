
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
from sklearn.manifold import TSNE

from tqdm.auto import tqdm

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
data = ravel(data_3d)
data.shape

# %%
tsne = TSNE(n_components=2, n_jobs=48)

d = tsne.fit_transform(data)
d.shape

# %%

np.save(os.path.join(data_folder, 'data_tsne2.npy'), d)

# %%

df = pd.DataFrame(columns=['x', 'y', 'e'])

df['e'] = event
df['e'] = df['e'].map(str)
df['x'] = d[:, 0]
df['y'] = d[:, 1]
df['o'] = range(len(df))
df

# %%

fig = px.scatter(df, x='x', y='y', color='e')
fig.write_html(os.path.join(data_folder, 'view_tsne2.html'))
fig.show()

# %%
