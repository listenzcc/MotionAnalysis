# %%
import os
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %%
pwd = os.path.dirname(__file__)
data_folder = os.path.join(pwd, '..', '..', 'data')

# Data_3d shape is (31625, 20, 6),
#                   samples x times x channels
# event shape is (31625,)
data_3d = np.load(os.path.join(data_folder, 'data3.npy'))
event = np.load(os.path.join(data_folder, 'event.npy'))

channel_name = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
event_idx = list(np.unique(event))
event_name = ['SLW', 'MLW', 'FLW', 'RD', 'SD', 'sit', 'stand', 'RA', 'SA']

data_3d.shape, event.shape

# %%


def select_batch(k, ravel=True):
    data_batch = []
    for e in event_idx:
        d = np.random.permutation(data_3d[event == e])[:k]
        m = np.mean(d, axis=0)
        if ravel:
            m = np.ravel(m)
        data_batch.append(m)
    return data_batch


def add_df(df, data_batch, k):
    for j, d in enumerate(data_batch):
        df = df.append(pd.Series(dict(
            feature=d,
            event=event_name[j],
            k=k
        )), ignore_index=True)

    df = df[['event', 'k', 'feature']]

    return df


# %%
df = pd.DataFrame()

for j in range(10):
    k = 2000
    data_batch = select_batch(k=k)
    df = add_df(df, data_batch, k)

for j in range(10):
    k = 1000
    data_batch = select_batch(k=k)
    df = add_df(df, data_batch, k)

for j in range(10):
    k = 500
    data_batch = select_batch(k=k)
    df = add_df(df, data_batch, k)

for j in range(100):
    k = 100
    data_batch = select_batch(k=k)
    df = add_df(df, data_batch, k)

df

# %%
d = np.array([e for e in df['feature']])

# pca = PCA(n_components=2)
# dp = pca.fit_transform(d)

tsne = TSNE(n_components=2)
dp = tsne.fit_transform(d)

d.shape, dp.shape

# %%

df['x'] = dp[:, 0]
df['y'] = dp[:, 1]
fig = px.scatter(df, x='x', y='y', color='event', size='k')
fig.show()

# %%
