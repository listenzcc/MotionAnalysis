# %%
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px

from tqdm.auto import tqdm

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %%
path = os.path.join(os.environ['WorkingFolder'],
                    'MotionAnalysis',
                    '处理后的数据(200ms)(6个人).csv')

# %%
df = pd.read_csv(path, header=None)
df

# %%
dataAll = df.values
label = dataAll[:, 0]
data = dataAll[:, 2:]

# %%
clf = make_pipeline(StandardScaler(), SVC(gamma='scale'))
skf = StratifiedKFold(n_splits=10, shuffle=True)

predict = label * 0

for train_idx, test_idx in tqdm(skf.split(data, label)):
    X = data[train_idx]
    y = label[train_idx]

    clf.fit(X, y)

    X1 = data[test_idx]
    y1 = label[test_idx]

    y2 = clf.predict(X1)

    print(metrics.classification_report(y1, y2))

    predict[test_idx] = y2


print(metrics.classification_report(label, predict))

# %%
rdf = pd.DataFrame(columns=['label', 'predict'])
rdf['label'] = label
rdf['predict'] = predict
rdf.to_csv(os.path.join(os.environ['WorkingFolder'],
                        'MotionAnalysis',
                        f'raw_classification_results-{time.time()}.csv'))

# %%
print(metrics.classification_report(label, predict))
cm = metrics.confusion_matrix(label, predict, normalize='true')
px.imshow(cm)

# %%
