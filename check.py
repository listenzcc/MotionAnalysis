# %%
import os
import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm

from sklearn import metrics

# %%
folder = os.path.join(os.environ['WorkingFolder'],
                      'MotionAnalysis')

# %%
dfs = []
for f in os.listdir(folder):
    if f.endswith('.csv') and f.startswith('raw_classification_results'):
        print(f)
        dfs.append(pd.read_csv(os.path.join(folder, f)))

dfs

# %%
for df in dfs:
    label = df['label']
    predict = df['predict']
    print(metrics.classification_report(label, predict))
    cm = metrics.confusion_matrix(label, predict, normalize='true')
    fig = px.imshow(cm)
    fig.show()

# %%
