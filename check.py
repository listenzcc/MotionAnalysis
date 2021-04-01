# %%
import os
import numpy as np
import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm

from sklearn import metrics

# %%
folder = os.path.join(os.path.dirname(__file__), '..', 'data')

# %%
df = pd.read_csv(os.path.join(folder, 'raw_classification_results.csv'))
df['predictCNN'] = np.load(os.path.join(folder, 'cnn_predict.npy'))

label = df['label']
predsvm = df['predict']
predcnn = df['predictCNN']

print('SVM')
print(metrics.classification_report(label, predsvm))
cm = metrics.confusion_matrix(label, predsvm, normalize='true')
fig = px.imshow(cm)
fig.show()

print('CNN')
print(metrics.classification_report(label, predcnn))
cm = metrics.confusion_matrix(label, predcnn, normalize='true')
fig = px.imshow(cm)
fig.show()
# %%
