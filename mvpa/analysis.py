# %%
import os
import numpy as np

import plotly.express as px

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# %%

event_name = {
    '0-SLW': 1,
    '1-MLW': 2,
    '2-FLW': 3,
    '3-RD': 4,
    '4-SD': 5,
    '5-sit': 6,
    '6-stand': 7,
    '7-RA': 8,
    '8-SA': 9,
}

folder_path = os.path.join(os.environ['SYNC'], 'MotionData\\data\\motions')


# %%
X_raw = np.zeros((0, 120 + 24))
Y_raw = np.zeros((0, 1))

kwargs = dict(axis=1, keepdims=True)

for name in event_name:
    x = np.load(os.path.join(folder_path, f'{name}.npy'))
    x_e = np.concatenate([
        np.max(x, **kwargs),
        np.min(x, **kwargs),
        np.mean(x, **kwargs),
        np.std(x, **kwargs),
    ], axis=1)
    print(x.shape, x_e.shape)
    y = event_name[name]

    # 120 = 20 x 6
    # 24 = 4 x 6
    X_raw = np.concatenate(
        [X_raw, np.concatenate([x.reshape(-1, 120), x_e.reshape(-1, 24)], axis=1)])
    Y_raw = np.concatenate([Y_raw, y + np.zeros((len(x), 1))])

    print(name, y, x.shape, X_raw.shape, Y_raw.shape)

Y_raw = Y_raw.reshape((len(Y_raw)))
print(X_raw.shape, np.unique(Y_raw))

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True)


def pipeline():
    return make_pipeline(StandardScaler(), SVC(gamma='scale'))


def fit_clf(X, y, events):
    print(f'{events}  ---------------------------------------------------')
    # Tell the [events] from others
    clf1 = pipeline()

    # Tell apart the [events]
    clf2 = pipeline()

    # Modify [y] for [clf1]
    yy = y.copy()
    y[[e in events for e in yy]] = 100
    y[[e not in events for e in yy]] = 200

    # Fit the [clf1] and [clf2]
    if len(np.unique(y)) > 1:
        clf1.fit(X, y)
    else:
        clf1 = None
    print(events, X.shape, y.shape, np.unique(y))
    clf2.fit(X[y == 100], yy[y == 100])
    print(events, X[y == 100].shape,
          yy[y == 100].shape, np.unique(yy[y == 100]))

    # Return
    # Discard the sample of [events]
    return X[y == 200], yy[y == 200], clf1, clf2


def select(X, y, events):
    s = [e in events for e in y]
    _y = y * 0 + 2
    _y[s] = 1
    return X[s], y[s], _y


for train_idx, test_idx in skf.split(Y_raw, Y_raw):
    print(train_idx.shape, test_idx.shape)
    # Train Data
    y_train = Y_raw[train_idx]
    X_train = X_raw[train_idx]

    # Test Data
    y_test = Y_raw[test_idx]
    X_test = X_raw[test_idx]

    clf = pipeline()
    clf.fit(X_train, y_train)

    # Fit on Steps
    X_train, y_train, clf_67_o, clf_67 = fit_clf(X_train, y_train, [6, 7])
    X_train, y_train, clf_123_o, clf_123 = fit_clf(X_train, y_train, [1, 2, 3])
    X_train, y_train, _, clf_4589 = fit_clf(X_train, y_train, [4, 5, 8, 9])
    print(X_train.shape, y_train.shape)

    break

# %%
y_pred = clf.predict(X_test)
print(np.unique(np.unique(y_pred)))
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

# %%
y_pred = clf_67_o.predict(X_test)
y_pred[y_pred == 100] = clf_67.predict(X_test[y_pred == 100])
y_pred[y_pred == 200] = clf_123_o.predict(X_test[y_pred == 200])
y_pred[y_pred == 100] = clf_123.predict(X_test[y_pred == 100])
y_pred[y_pred == 200] = clf.predict(X_test[y_pred == 200])

print(np.unique(np.unique(y_pred)))
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))


# %%
y_pred = clf_67_o.predict(X_test)
y_pred[y_pred == 100] = clf_67.predict(X_test[y_pred == 100])
y_pred[y_pred == 200] = clf.predict(X_test[y_pred == 200])

print(np.unique(np.unique(y_pred)))
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

# %%
