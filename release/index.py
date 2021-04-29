
'''
# Establish Dash App

## Information

- FileName: index.py
- Author: Chuncheng Zhang
- Date: 2021-04-27

## Script Function

The Index App for Visualize the Motion Trace.

The script will establish the web server,
the user can use it as the common web application.
The web server is established using the dash app.

The app is an interactive application,
the user can choose the motion event to be visualized,
the user can also watch the motion animation in one-by-one frame manner.
'''

# %%
import os

import numpy as np
import pandas as pd
import plotly.express as px

import matplotlib
import matplotlib.pyplot as plt

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

# %%
event_name = {
    "1-SLW": 1,
    "2-MLW": 2,
    "3-FLW": 3,
    "4-RD": 4,
    "5-SD": 5,
    "6-sit": 6,
    "7-stand": 7,
    "8-RA": 8,
    "9-SA": 9,
}
default_name = [e for e in event_name][0]

# %%
folder = os.path.join(os.environ['Sync'], 'MotionData', 'dataFolder')


def generate_path(last, folder=folder):
    '''
    Generate path based on [folder] and [last],
    the [last] will be tailed to the [folder] string.
    It will raise a warning message if the new path has been already a file.

    Args:
    - @last: The last part of the new path.
             If it is a list, it will be jointed together in the given order;
    - @folder: The folder part of the new path, it has default value.

    Outputs:
    - @output: The generated path string.
    '''

    if isinstance(last, list):
        last = os.path.sep.join(last)

    output = os.path.join(folder, last)

    if os.path.isfile(output):
        print(f'W: File exits: "{output}"')

    return output


def save(method, path, *args, **kwargs):
    '''
    Save to [path] using [method].

    Args:
    - @method: The method of saving;
    - @path: The path of saving.
    '''
    if os.path.isfile(path):
        print(f'W: Overriding existing file: "{path}"')

    method(path, *args, **kwargs)
    a = f'{args}'[:10]
    b = f'{kwargs}'[:10]
    print(f'I: Saved file: "{path}", with parameters of {a} and {b}')


# %%
'''
Compute triangle functions based on angle.
The angle is in the unit of degrees.
'''


def sin(x):
    '''
    Compute the sin value of angle [x].

    Args:
    - @x: The angle to be computed, the unit is degree.

    Outputs:
    - The sin value of [x].
    '''
    return np.sin(x / 180 * np.pi)


def cos(x):
    '''
    Compute the cos value of angle [x].

    Args:
    - @x: The angle to be computed, the unit is degree.

    Outputs:
    - The cos value of [x].
    '''
    return np.cos(x / 180 * np.pi)


def rgb(i, n=20):
    '''
    Generate color in hex format, like "#FFFFFF".
    Generate rgb hex string for colors in colormap of 'cool'.

    Args:
    - @i: The index of the color;
    - @n: The max of the color index, default by 20.
    '''
    c = plt.cm.cool(i/n)
    return matplotlib.colors.to_hex(c)


def rotate(rot):
    '''
    The rotation in 3D space can be expressed as the location multiples the rotation matrix,
    the function is used to compute the matrix.
    The rotation matrix has three sub-matrices:
    - The sub matrix of x-axis;
    - The sub matrix of y-axis;
    - The sub matrix of z-axis;
    - The three rotations are independent.

    Args:
    - @rot: The rotation vector,
            the three values are the angle of x-, y-, and z-axis,
            the unit is degree.

    Outputs:
    - @mat: The rotation matrix.
    '''
    # Parse the rotation angles
    rx, ry, rz = rot

    # The sub matrix of z-axis
    mz = np.array([[cos(rz), sin(rz), 0],
                   [-sin(rz), cos(rz), 0],
                   [0, 0, 1]])

    # The sub matrix of y-axis
    my = np.array([[cos(ry), 0, -sin(ry)],
                   [0, 1, 0],
                   [sin(ry), 0, cos(ry)]])

    # The sub matrix of x-axis
    mx = np.array([[1, 0, 0],
                   [0, cos(rx), sin(rx)],
                   [0, -sin(rx), cos(rx)]])

    # The rotation matrix is computed as mz x my x mx
    mat = np.matmul(np.matmul(mz, my), mx)

    return mat


# %%


class Trace(object):
    '''
    The Trace of the robs.

    The purpose of the trace is tracing the rigid transformation of the rod.
        The tracing is saved in [self.df],
        each time point is a two-rows pair in the [self.df],
        - The 1st row is the start point of the rod;
        - The 2nd row is the end point of the rod;
        - The columns are ['x', 'y', 'z', 'color', 'step'],
          refer the x-, y-, and z-axis positions, its color and step index.
    '''

    def __init__(self, vec, pos=[0, 0, 0], rot=[0, 0, 0], color='black'):
        '''
        Initialize the Trace.

        Args:
        - @vec: The rob's vector;
        - @pos: The origin coordinates of the [vec];
        - @rot: The rotation of the [vec];
        - @color: The color of the [vec].

        Outputs:
        - @self.df: The trace will be stored in the DataFrame,
                    the initial trace entry will be added as initialization.
        '''
        typ = np.float64
        self.vec = np.array(vec).astype(typ)
        self.pos = np.array(pos).astype(typ)
        self.rot = np.array(rot).astype(typ)

        self.step = 0

        rod = self.place()
        rod['color'] = [color for _ in range(len(rod))]
        rod['step'] = 0

        self.df = rod

    def place(self):
        '''
        The inner function of adding the latest trace.
        It is automatically called after [self.add] operation.
        It is extremely unrecommended to use this function by users.

        Outputs:
        - @rod: The latest trace.
        '''
        new_vec = np.matmul(self.vec, rotate(self.rot))
        p2 = self.pos + new_vec

        rod = pd.DataFrame(np.array([self.pos, p2]))
        rod.columns = ['x', 'y', 'z']

        return rod

    def add(self, pos_diff=[0, 0, 0], rot_diff=[0, 0, 0], color='black'):
        '''
        Add latest trace by the parameters.

        Args:
        - @pos_diff: The different position in x-, y-, and z-axis;
        - @rot_diff: The different rotation in x-, y-, and z-axis in degree;
        - @color: The color of the rob.

        Generate:
        - The newly added entry is added into the [self.df].
        '''
        typ = np.float64
        self.pos += np.array(pos_diff).astype(typ)
        self.rot += np.array(rot_diff).astype(typ)
        self.step += 1

        rod = self.place()
        rod['color'] = [color for _ in range(len(rod))]
        rod['step'] = self.step

        self.df = pd.concat([self.df, rod], axis=0)


# %%
'''
Plot Animation
'''

length = 10
vec = [-length, 0, 0]


def plot_trace(e, dd, vec=vec):
    '''
    Plot the trace based on the data [dd] for the motion event [e].
    The trace of the motion is computed,
    and the motion is drawn in animation.

    Args:
    - @e: The event name;
    - @dd: The data in 3D matrix;
    - @vec: The initial vector of the rob.

    Outputs:
    - @fig: The figure of the animation;
    - @trace.df: The trace of the motion.
    '''
    trace = Trace(vec, color=rgb(0))

    for i, d in enumerate(dd):
        trace.add(pos_diff=d[:3], rot_diff=d[3:], color=rgb(i))

    trace.df

    range_x = (trace.df['x'].min(), trace.df['x'].max())
    range_y = (trace.df['y'].min(), trace.df['y'].max())
    range_z = (trace.df['z'].min(), trace.df['z'].max())

    df1 = trace.df.copy()
    df1['line_group'] = 'a'

    # Add Grids
    dfx = trace.df.copy()
    dfx['line_group'] = 'g'

    dfy = trace.df.copy()
    dfy['line_group'] = 'g'

    dfz = trace.df.copy()
    dfz['line_group'] = 'g'

    # # x-axis
    # xyz = dfx[['x', 'y', 'z']].values
    # for j in range(len(xyz)):
    #     if j % 2 == 0:
    #         xyz[j] = [range_x[0], range_y[1], range_z[1]]
    #     if j % 2 == 1:
    #         xyz[j] = [range_x[1], range_y[1], range_z[1]]
    # dfx[['x', 'y', 'z']] = xyz

    # # y-axis
    # xyz = dfy[['x', 'y', 'z']].values
    # for j in range(len(xyz)):
    #     if j % 2 == 0:
    #         xyz[j] = [range_x[0], range_y[1], range_z[1]]
    #     if j % 2 == 1:
    #         xyz[j] = [range_x[0], range_y[0], range_z[1]]
    # dfy[['x', 'y', 'z']] = xyz

    # # z-axis
    # xyz = dfz[['x', 'y', 'z']].values
    # for j in range(len(xyz)):
    #     if j % 2 == 0:
    #         xyz[j] = [range_x[0], range_y[1], range_z[1]]
    #     if j % 2 == 1:
    #         xyz[j] = [range_x[0], range_y[1], range_z[0]]
    # dfz[['x', 'y', 'z']] = xyz

    # df = pd.concat([df1, dfx, dfy, dfz], axis=0)
    df = df1.copy()

    kwargs = dict(
        width=800,
        height=800,
        title=e
    )

    fig = px.line_3d(df, x='x', y='y', z='z', line_group='line_group',
                     color='color', **kwargs)

    for j, d in enumerate(fig.data):
        d['line']['color'] = df.iloc[j*2]['color']

    return fig, trace.df


# %%
'''
Set up the camera parameters,
- @up: The up direction of the fig;
- @center: The center position of the fig;
- @eye: The eys position of the observer.
'''

camera = dict(
    up=dict(x=1, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=-2, z=0)
)


class MyFigure(object):
    '''
    The Figure Manager.
    The purpose of the object is to build a container for the figure,
    it can maintain the figure for the useage of the dash app,
    the manager can make the figure changeable,
    it is essential to build the interactive app.
    '''

    def __init__(self, name):
        '''
        Initialize the object by the motion name [name].

        Args:
        - @name: The motion name of interest.
        '''
        self.update_fig(name)

    def update_fig(self, name):
        '''
        Update the figure by the motion name [name].

        Args:
        - @name: The motion name of interest.

        Outputs:
        - @fig: The generated figure.
        '''
        d = np.load(generate_path(f'{name}.npy'))
        md = np.mean(d, axis=0)
        fig, _ = plot_trace(name, md)
        fig.update_layout(scene_camera=camera)
        self.fig = fig
        self.step = 0
        print(f'I: Changed figure to the motion of "{name}"')
        return fig

    def forward(self):
        '''
        Move forward the step to the current figure.

        Outputs:
        - @fig: The current figure.
        '''
        n = len(self.fig.data)

        for d in self.fig.data:
            d['line']['width'] = 5
        d = self.fig.data[self.step]

        d['line']['width'] = 20

        self.step += 1
        self.step %= n

        print(f'D: Changed step into "{self.step}"')
        return self.fig


# Init the MyFigure instance
myfig = MyFigure(default_name)

# %%
'''
Generate and Layout the Dash App
'''
# Main App Instance
app = dash.Dash(__name__)

# Component Style Parameters
style = {'width': '100%', 'display': 'inline-block'}

# Setup App Layout
app.layout = html.Div(
    [
        # Motion Event Selector
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown-1',
                    options=[{'label': e, 'value': e} for e in event_name],
                    value=default_name,
                )
            ],
            style=style
        ),
        # Main Graph
        html.Div(
            [
                dcc.Graph(id='graph-1', figure=myfig.fig)
            ],
            style=style
        ),
        # Button Container
        html.Div(
            [
                html.Button('Forward', id='button-1', n_clicks=0)
            ],
            style=style
        ),
    ]
)


# Setup callback for the dropdown menu and button
@app.callback(
    Output(component_id='graph-1', component_property='figure'),
    [
        Input(component_id='dropdown-1', component_property='value'),
        Input(component_id='button-1', component_property='n_clicks'),
    ]
)
def update_graph1(name, n_clicks1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(f'D: Component: "{changed_id}" changed.')

    if changed_id.startswith('dropdown-1'):
        myfig.update_fig(name)

    if changed_id.startswith('button-1'):
        myfig.forward()

    return myfig.fig


# %%
# Start up the App on running the Script
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
