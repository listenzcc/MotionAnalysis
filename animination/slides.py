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
    num_range=(10, 20),
    num_frames=20,
)

# %%
np.random.choice([1, 2, 3])

# %%
# Make frames
frame_df = []


def random(rge):
    bot, top = rge
    return np.random.randint(bot, top)


for j in tqdm(range(setting['num_frames'])):
    df = pd.DataFrame()

    num = random(setting['num_range'])
    df['x'] = [random(setting['x_range']) for _ in range(num)]
    df['y'] = [random(setting['y_range']) for _ in range(num)]
    df['color'] = np.random.choice(setting['color'], num)
    df['size'] = np.random.choice(setting['size'], num)
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
