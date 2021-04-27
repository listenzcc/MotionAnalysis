
# %%
import os
import numpy as np

import plotly.express as px

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(144, 80)
        self.l2 = torch.nn.Linear(80, 3)

    def forward(self, x, dropout=0.2):
        x = F.relu(self.l1(x))
        x = F.dropout(x, p=dropout)
        # x = torch.sigmoid(self.l2(x))
        x = F.relu(self.l2(x))
        return x

# %%


def pipeline():
    return make_pipeline(StandardScaler(), SVC(gamma='scale'))


# %%
skf = StratifiedKFold(n_splits=5, shuffle=True)


def select(X, y, events):
    s = [e in events for e in y]
    return X[s], y[s]


for train_idx, test_idx in skf.split(Y_raw, Y_raw):
    print(train_idx.shape, test_idx.shape)
    # Train Data
    y_train = Y_raw[train_idx]
    X_train = X_raw[train_idx]

    # Test Data
    y_test = Y_raw[test_idx]
    X_test = X_raw[test_idx]

    events = [1, 2, 3]
    X_train, y_train = select(X_train, y_train, events)
    X_test, y_test = select(X_test, y_test, events)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(X_train.shape, X_test.shape)

    clf = pipeline()
    clf.fit(X_train, y_train)

    model = Net().cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.2)

    for epoch_idx in range(4000):
        skf_inner = StratifiedKFold(n_splits=10, shuffle=True)
        for _, idx in skf_inner.split(y_train, y_train):
            X = X_train[idx]
            # y = y_train[idx][:, np.newaxis] - 1
            y = y_train[idx] - 1

            # y = np.zeros((len(y_raw), 3))
            # for i, _ in enumerate(y):
            #     y[i, int(y_raw[i]-1)] = 1

            # print(X.shape, y.shape)

            optimizer.zero_grad()
            outputs = model(torch.Tensor(X).cuda())
            loss = loss_fn(outputs, torch.Tensor(y).cuda().long())
            loss.backward()
            optimizer.step()

        if epoch_idx % 50 == 0:
            print(epoch_idx, loss.item())

        if epoch_idx == 2000:
            optimizer.param_groups[0]['lr'] = 0.005

    break

# %%
pred = model(torch.Tensor(X_test).cuda(), dropout=0)
pred = pred.cpu().detach().numpy()

y_pred = np.zeros(len(pred))
for j in range(len(pred)):
    y_pred[j] = np.argmax(pred[j]) + 1

print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

# %%
y_pred1 = clf.predict(X_test)
print(metrics.classification_report(y_true=y_test, y_pred=y_pred1))
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred1))

# %%
