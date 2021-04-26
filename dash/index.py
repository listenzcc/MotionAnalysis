# %%
import os
import dash
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import matplotlib
import matplotlib.pyplot as plt
# %%
'''
Compute triangle functions based on angle.
The angle is in the unit of degrees.
'''


def sin(x):
    '''
    Compute sin of [x] in the unit of degrees.

    Args:
    - @x: The angle value in degrees.
    '''
    return np.sin(x / 180 * np.pi)


def cos(x):
    '''
    Compute cos of [x] in the unit of degrees.

    Args:
    - @x: The angle value in degrees.
    '''
    return np.cos(x / 180 * np.pi)


def rotate(rot):
    '''
    Compute rotate matrix for [rot].
    The matrix is designed to be right-multipled to the vector to perform the rotation.

    Args:
    - @rot: The rotation's 3 params, the unit is degree.
    '''
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
'''
Generate rgb hex string for colors in colormap of 'cool'
'''


def rgb(i, n=20):
    '''
    Generate color in hex format, like "#FFFFFF".

    Args:
    - @i: The index of the color;
    - @n: The max of the color index, default by 20.
    '''
    c = plt.cm.cool(i/n)
    return matplotlib.colors.to_hex(c)

# %%


class Trace(object):
    ''' Trace Motion and Manage the Positions of the Rod '''

    def __init__(self, vec, pos=[0, 0, 0], rot=[0, 0, 0], color='black'):
        '''
        Init the Trace.

        The purpose of the trace is tracing the rigid transformation of the rod.
        The tracing is saved in [self.df],
        the columns are ['x', 'y', 'z', 'color', 'step'],
        each trace has two rows:
          1. The start end of the rod;
          2. The stop end of the rod.

        The [vec] refers the origin rod position (without rigid transformation),
        and the [pos] / [rot] params refer the 6-value rigid transformation.

        Args:
        - @vec: The initial vector of the rod;
        - @pos: The current position of the rod, default by [0, 0, 0], unit is unknown;
        - @rot: The rotation of the rod, default by [0, 0, 0], unit is degree;
        - @color: The color of the rod, default by 'black'.
        '''

        self.vec = np.array(vec).astype(np.float)
        self.pos = np.array(pos).astype(np.float)
        self.rot = np.array(rot).astype(np.float)

        self.step = 0

        rod = self._place()
        rod['color'] = [color for _ in range(len(rod))]
        rod['step'] = 0

        self.df = rod

    def _place(self):
        '''
        Compute the position of the rod with rigid transformation
        '''

        new_vec = np.matmul(self.vec, rotate(self.rot))
        p2 = self.pos + new_vec

        rod = pd.DataFrame(np.array([self.pos, p2]))
        rod.columns = ['x', 'y', 'z']

        return rod

    def add(self, pos_diff=[0, 0, 0], rot_diff=[0, 0, 0], color='black'):
        '''
        Add new trace of the rod,
        the new trace will be added to the [self.df].

        Args:
        - @pos_diff: The latest change of the position's 3 params, default by [0, 0, 0];
        - @rot_diff: The latest change of the rotation's 3 params, default by [0, 0, 0];
        - @color: The color of the rod, default by 'black'.
        '''

        self.pos += np.array(pos_diff).astype(np.float)
        self.rot += np.array(rot_diff).astype(np.float)
        self.step += 1

        rod = self._place()
        rod['color'] = [color for _ in range(len(rod))]
        rod['step'] = self.step

        self.df = pd.concat([self.df, rod], axis=0)


# %%
'''
Plot Animation
'''

length = 10
vec = [-length, 0, 0]
camera = dict(
    up=dict(x=1, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=-2, z=0)
)


def plot_trace(e, dd, vec=vec, camera=camera):
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
    fig.update_layout(scene_camera=camera)

    return fig


# %%
app = dash.Dash(__name__)
folder = r'H:\Sync\MotionData\data\motions'
names = ['0-SLW', '1-MLW', '2-FLW', '3-RD',
         '4-SD', '5-sit', '6-stand', '7-RA', '8-SA']


app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Dropdown(
                    id='motionName',
                    options=[{'label': i, 'value': i} for i in names],
                    value=names[0],
                    style={'width': '100%'}
                ),
            ],
            style={'width': '100%', 'display': 'inline-block'}
        ),
        html.Div(
            [
                dcc.Graph(id='graph1')
            ],
            style={'width': '100%', 'display': 'inline-block'}
        ),
    ]
)


@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [
        Input(component_id='motionName', component_property='value'),
    ]
)
def update_output(name):
    data = np.load(os.path.join(folder, f'{name}.npy'))
    mean_data = np.mean(data, axis=0)
    print(mean_data.shape)

    fig = plot_trace(name, mean_data)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
