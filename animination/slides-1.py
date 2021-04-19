
# %%
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from PIL import Image

# %%


def getPixels_1(filename):
    print(filename)
    t = time.time()
    img = Image.open(filename, 'r')
    w, h = img.size
    mat = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    print(f'Reading {filename} costs {time.time() - t} seconds')
    return mat


def getPixels(filename, sz=(1000, 1000, 3)):
    print(filename)
    t = time.time()
    mat = plt.imread(filename)
    shape = mat.shape
    print(shape)
    a = np.linspace(0, shape[0], sz[0], endpoint=False).astype(np.int)
    b = np.linspace(0, shape[1], sz[1], endpoint=False).astype(np.int)
    mat = mat[a][:, b][:, :, :sz[2]]
    if not mat.dtype == np.uint8:
        print(
            f'Warning: converting the {filename} into uint8 format, it can be wrong, check it.')
        mat *= 255
        mat = mat.astype(np.uint8)
    print(f'Reading {filename} ({mat.shape}) costs {time.time() - t} seconds.')
    return mat


# %%
folder = os.path.join(os.environ['ONEDRIVE'], 'Pictures', 'DesktopPictures')
# pics = [getPixels(os.path.join(folder, f)) for f in os.listdir(folder)]
names = [f for f in os.listdir(folder)]
pics = [getPixels(os.path.join(folder, f)) for f in names]

# %%
data = np.array(pics)
print(data.shape)
fig = px.imshow(data, animation_frame=0)
for j, n in enumerate(names):
    fig.layout['sliders'][0]['steps'][j]['label'] = n
fig.write_html('b_tmp.html')
fig.show()


# %%
