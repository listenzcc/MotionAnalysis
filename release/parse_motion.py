'''
# Parse Motion Events

## Information

- FileName: parse_motion.py
- Author: Chuncheng Zhang
- Date: 2021-04-27

## Dependency

The script is designed to be operated **AFTER** running the script of "parse_raw_data.py".
It will use the data of the 9 motion events.
- 1-SLW: Slow-speed Level Walking;
- 2-MLW: Medium-speed Level Walking;
- 3-FLW: Fast-speed Level Walking;
- 4-RD: Ramp Descending;
- 5-SD: Stair Descending;
- 6-sit: Sitting Down;
- 7-stand: Standing Up;
- 8-RA: Ramp Ascending;
- 9-SA: Stair Ascending.

## Analysis

The script will generate the trace and the animation of the motion events.
'''
# %%
import os

import numpy as np
import pandas as pd
import plotly.express as px

import matplotlib
import matplotlib.pyplot as plt

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
        the columns are ['x', 'y', 'z', 'color', 'step'],
        each trace has two rows:
          1. The start end of the rod;
          2. The stop end of the rod.
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


def plot_animation(e, dd, vec=vec):
    '''
    Plot the animation based on the data [dd] for the motion event [e].
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

    # x-axis
    xyz = dfx[['x', 'y', 'z']].values
    for j in range(len(xyz)):
        if j % 2 == 0:
            xyz[j] = [range_x[0], range_y[1], range_z[1]]
        if j % 2 == 1:
            xyz[j] = [range_x[1], range_y[1], range_z[1]]
    dfx[['x', 'y', 'z']] = xyz

    # y-axis
    xyz = dfy[['x', 'y', 'z']].values
    for j in range(len(xyz)):
        if j % 2 == 0:
            xyz[j] = [range_x[0], range_y[1], range_z[1]]
        if j % 2 == 1:
            xyz[j] = [range_x[0], range_y[0], range_z[1]]
    dfy[['x', 'y', 'z']] = xyz

    # z-axis
    xyz = dfz[['x', 'y', 'z']].values
    for j in range(len(xyz)):
        if j % 2 == 0:
            xyz[j] = [range_x[0], range_y[1], range_z[1]]
        if j % 2 == 1:
            xyz[j] = [range_x[0], range_y[1], range_z[0]]
    dfz[['x', 'y', 'z']] = xyz

    df = pd.concat([df1, dfx, dfy, dfz], axis=0)

    kwargs = dict(
        width=800,
        height=800,
        title=e
    )

    fig = px.line_3d(df, x='x', y='y', z='z', line_group='line_group',
                     color='color', animation_frame='step', **kwargs)

    fig_s = px.line_3d(df, x='x', y='y', z='z', line_group='line_group',
                       color='color', **kwargs)

    for f in fig.frames:
        f.data[0].name = 'x'

    for j, frame in enumerate(fig.frames):
        frame['data'][0]['line']['color'] = df.iloc[j*2]['color']
        frame['data'][0]['line']['width'] = 5
        frame['data'][1]['line']['color'] = 'black'

    for j, d in enumerate(fig_s.data):
        d['line']['color'] = df.iloc[j*2]['color']

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 20

    return fig, fig_s, trace.df


# %%
'''
Draw the animation plot
'''
camera = dict(
    up=dict(x=1, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=-2, z=0)
)

for name in event_name:
    print(name)

    d = np.load(generate_path(f'{name}.npy'))
    md = np.mean(d, axis=0)
    print(name, d.shape, md.shape)

    fig, fig_s, _ = plot_animation(name, md)
    fig.update_layout(scene_camera=camera)
    fig_s.update_layout(scene_camera=camera)
    save(fig.write_html, generate_path(f'{name}.html'))
    save(fig_s.write_html, generate_path(f'{name}_stable.html'))

# %%
