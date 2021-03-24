
# %%
import os
import numpy as np
import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm

# %%
channel_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
folder = os.path.join(os.environ['WorkingFolder'],
                      'MotionAnalysis')
path = os.path.join(folder,
                    '处理后的数据(200ms)(6个人).csv')

# %%
df = pd.read_csv(path, header=None)
df

# %%
dataAll = df.values
event = dataAll[:, 0]
data = dataAll[:, 2:]

# %%
s = data.shape
ns = (data.shape[0], int(data.shape[1]/6), 6)
data3 = np.zeros(ns)
for i in range(s[0]):
    data3[i] = data[i].reshape((ns[1], ns[2]))
data3.shape

np.save(os.path.join(folder, 'data3.npy'), data3)
np.save(os.path.join(folder, 'event.npy'), event)
# %%
frame = pd.DataFrame()
data32 = np.concatenate([data3[:, i, :] for i in range(20)], axis=0)
event32 = np.concatenate([event for _ in range(20)], axis=0)
time32 = np.concatenate([event * 0 + i for i in range(20)], axis=0)

frame['event'] = event32
frame['time'] = time32
frame[channel_names] = data32
frame

# %%
frame

# %%
stat_frame = pd.DataFrame()
for e in frame['event'].unique():
    for t in frame['time'].unique():
        mean = frame.query(f'event == {e}').query(f'time == {t}').mean(axis=0)
        mean['event'] = e
        mean['time'] = t
        stat_frame = stat_frame.append(mean, ignore_index=True)

stat_frame = stat_frame[frame.columns]
stat_frame

# %%
html = open('template.html').read()

include_plotlyjs = True

for c in channel_names:
    fig = px.line(stat_frame, title=c, y=c, x='time', color='event')
    # fig.show()
    html = html.replace(f'{{{{{c}}}}}',
                        fig.to_html(full_html=False,
                                    include_plotlyjs=include_plotlyjs).replace('div', 'div class="subplot"', 1))
    include_plotlyjs = False

with open(os.path.join(folder, 'waveform.html'), 'w') as f:
    f.write(html)

# %%
