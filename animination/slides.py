''' Build animation for scatters '''

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from tqdm.auto import tqdm

# %%
setting = dict(
    size=[10, 20, 30],
    color=['hsl(0, 100%, 50%)', 'hsl(70, 100%, 50%)', 'hsl(130, 100%, 50%)'],
    x_range=(10, 80),
    y_range=(10, 20),
    num=10,
    num_frames=20,
)

# %%
np.random.choice([1, 2, 3])

# %%
# Make frames
frame_df = []

for j in tqdm(range(setting['num_frames'])):

    df = pd.DataFrame()
    df['x'] = [np.random.randint(setting['x_range'][0], setting['x_range'][1])
               for _ in range(setting['num'])]
    df['y'] = [np.random.randint(setting['y_range'][0], setting['y_range'][1])
               for _ in range(setting['num'])]
    df['color'] = np.random.choice(setting['color'], setting['num'])
    df['size'] = np.random.choice(setting['size'], setting['num'])
    df['name'] = f'Frame-{j:02d}'

    frame_df.append(df)

frame_df = pd.concat(frame_df, axis=0)
frame_df

# %%
df = px.data.gapminder()
df

# %%
fig = px.scatter(frame_df, x='x', y='y', color='color',
                 size='size', animation_frame='name')
fig.write_html('a_tmp.html')
fig.show()

# %%
