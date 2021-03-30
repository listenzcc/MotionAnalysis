# View in 3D

# %%
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# %%

# ----------------------------------------------------
# Compute triangle functions based on angle.
# The angle is in the unit of degrees.
# ----------------------------------------------------


def sin(x):
    return np.sin(x / 180 * np.pi)


def cos(x):
    return np.cos(x / 180 * np.pi)


def rotate(rot):
    rx, ry, rz = rot

    mz = np.array([[cos(rz), sin(rz), 0],
                   [-sin(rz), cos(rz), 0],
                   [0, 0, 1]])

    my = np.array([[cos(ry), 0, -sin(ry)],
                   [0, 1, 0],
                   [sin(ry), 0, cos(ry)]])

    mx = np.array([[1, 0, 0],
                   [0, cos(rx), sin(rx)],
                   [0, -sin(rx), cos(rx)]])

    mat = np.matmul(np.matmul(mz, my), mx)

    return mat


# %%


class Trace(object):
    def __init__(self, vec, pos=[0, 0, 0], rot=[0, 0, 0], color='black'):
        self.vec = np.array(vec).astype(np.float)
        self.pos = np.array(pos).astype(np.float)
        self.rot = np.array(rot).astype(np.float)

        self.step = 0

        rod = self.place()
        rod['color'] = [color for _ in range(len(rod))]
        rod['step'] = 0

        self.df = rod

    def place(self):
        new_vec = np.matmul(self.vec, rotate(self.rot))
        p2 = self.pos + new_vec

        rod = pd.DataFrame(np.array([self.pos, p2]))
        rod.columns = ['x', 'y', 'z']

        return rod

    def add(self, pos_diff=[0, 0, 0], rot_diff=[0, 0, 0], color='black'):
        self.pos += np.array(pos_diff).astype(np.float)
        self.rot += np.array(rot_diff).astype(np.float)
        self.step += 1

        rod = self.place()
        rod['color'] = [color for _ in range(len(rod))]
        rod['step'] = self.step

        self.df = pd.concat([self.df, rod], axis=0)


# %%
data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Data_3d shape is (31625, 20, 6),
#                   samples x times x channels
# event shape is (31625,)
data_3d = np.load(os.path.join(data_folder, 'data3.npy'))
event = np.load(os.path.join(data_folder, 'event.npy'))

channel_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

data_3d.shape, event.shape

# %%
mean_data = []

for e in np.unique(event):
    mean_data.append([e, np.mean(data_3d[event == e], axis=0)])

# %%


def rgb(i, n=20):
    return plt.cm.cool(i/n)


length = 10
vec = [0, 0, -length]

html = open('template_motion_animation.html').read()
include_plotlyjs = True

for md in mean_data:
    e, dd = md
    e = int(e)
    print(f'Event of {e}')

    trace = Trace(vec, color=rgb(0))

    for i, d in enumerate(dd):
        trace.add(pos_diff=d[:3], rot_diff=d[3:], color=rgb(i))

    data = []
    layout = dict(
        title=e,
    )

    for i in trace.df['step'].unique():
        df = trace.df.query(f'step=={i}')

        d = go.Scatter3d(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            marker=dict(
                size=2,
                color=df['color'],
            ),
            line=dict(
                color=df['color']
            )
        )

        data.append(d)

    fig = go.Figure(data=data, layout=layout)
    # fig.show()

    html = html.replace(f'{{{{event {e}}}}}',
                        fig.to_html(full_html=False,
                                    include_plotlyjs=include_plotlyjs))
    print(len(html))
    include_plotlyjs = False


# %%

with open(os.path.join(data_folder, 'motion_animation.html'), 'w') as f:
    f.write(html)

# %%
