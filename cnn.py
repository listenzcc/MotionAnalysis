# %%
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# %%

data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

# %%
# Data shape is (31625, 20, 6),
#                samples x times x channels
# event shape is (31625,)
data = np.load(os.path.join(data_folder, 'data3.npy'))
event = np.load(os.path.join(data_folder, 'event.npy'))
data.shape, event.shape

m, n = np.max(data), np.min(data)
data = (data - n) / (m - n) - 0.5

# %%


def onehot_encode(y, n=9):
    assert(len(y.shape) == 1)
    m = len(y)
    yy = np.zeros((m, n))
    for i, e in enumerate(y):
        yy[i][int(e-1)] = 1
    return yy


def onehot_decode(yy):
    assert(len(yy.shape) == 2)
    m = len(yy)
    y = np.zeros((m))
    for i, e in enumerate(yy):
        y[i] = np.where(e == max(e))[0][0] + 1
    return y

# %%


settings = dict(
    times=20,
    times_kernel_num=6,
    times_kernel_width=1,
    times_padding=0,
    channels=6,
    channels_feature_num=30,
    channels_kernel_width=6,
    fc1_input_num=600,
    fc2_input_num=200,
    fc2_output_num=9,
)

DEVICE = 'cuda'


def numpy2torch(array, dtype=np.float32, device=DEVICE):
    # Attach [array] to the type of torch
    return torch.from_numpy(array.astype(dtype)).to(device)


def torch2numpy(tensor):
    # Detach [tensor] to the type of numpy
    return tensor.detach().cpu().numpy()


class CNNNet(torch.nn.Module):
    # CNN Net of 3 layers,
    # input shape is (n x times x channels),
    # n is the number of samples
    def __init__(self):
        super(CNNNet, self).__init__()
        self.layout()

    def layout(self):
        self.conv1 = nn.Conv2d(1,
                               settings['times_kernel_num'],
                               (settings['times_kernel_width'], 1),
                               padding=[settings['times_padding'], 0])

        self.batchnorm1 = nn.BatchNorm2d(settings['times_kernel_num'],
                                         False)

        self.conv2 = nn.Conv2d(settings['times_kernel_num'],
                               settings['channels_feature_num'],
                               (1, settings['channels_kernel_width']),
                               padding=[0, 0])

        self.fc1 = nn.Linear(settings['fc1_input_num'],
                             settings['fc2_input_num'])

        self.fc2 = nn.Linear(settings['fc2_input_num'],
                             settings['fc2_output_num'])

    def max_norm(self):
        # def max_norm(model, max_val=2, eps=1e-8):
        #     for name, param in model.named_parameters():
        #         if 'bias' not in name:
        #             norm = param.norm(2, dim=0, keepdim=True)
        #             # desired = torch.clamp(norm, 0, max_val)
        #             param = torch.clamp(norm, 0, max_val)
        #             # param = param * (desired / (eps + norm))
        eps = 1e-8
        for name, param in self.named_parameters():
            if 'bias' in name:
                continue

            max_val = 2
            if any([name.startswith(e) for e in ['conv']]):
                norm = param.norm(2, dim=None, keepdim=True)
                desired = torch.clamp(norm, 0, max_val)
                param = param * desired / (eps + norm)
                continue

            max_val = 0.5
            if any([name.startswith(e) for e in ['fc']]):
                norm = param.norm(2, dim=None, keepdim=True)
                desired = torch.clamp(norm, 0, max_val)
                param = param * desired / (eps + norm)
                continue

    def forward(self, x, dropout=0.25):
        x = self.conv1(x)
        # x = self.batchnorm1(x)
        x = F.elu(x)
        x = F.dropout(x, dropout)

        x = self.conv2(x)
        x = F.elu(x)
        x = F.dropout(x, dropout)

        x = x.view(-1, settings['fc1_input_num'])
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


net = CNNNet().cuda()
torchsummary.summary(net, input_size=(1, 20, 6))

# %%

predict = event * 0
predict_svm = event * 0


clf = make_pipeline(StandardScaler(), SVC(gamma='scale'))
epochs_num = 50

outer_skf = StratifiedKFold(n_splits=10, shuffle=True)

for train_idx, test_idx in outer_skf.split(data, event):
    net = CNNNet().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),
                           lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=50,
                                          gamma=0.9)

    for count in range(500):
        inner_skf = StratifiedKFold(n_splits=epochs_num, shuffle=True)
        for _, epochs in inner_skf.split(data[train_idx], event[train_idx]):

            raw_x = data[train_idx][epochs]
            raw_y = event[train_idx][epochs]

            x = numpy2torch(raw_x[:, np.newaxis, :, :])
            yy = numpy2torch(onehot_encode(raw_y))

            optimizer.zero_grad()
            pp = net.forward(x)

            loss = criterion(pp, yy)
            loss.backward()

            optimizer.step()
            net.max_norm()

        print(f'{count:04d} Loss: {loss.item():.6f}')
        scheduler.step()

    x = numpy2torch(data[test_idx][:, np.newaxis, :, :])
    p = onehot_decode(torch2numpy(net.forward(x, dropout=0)))
    predict[test_idx] = p

    print(metrics.classification_report(y_true=event[test_idx], y_pred=p))


# %%
print(metrics.classification_report(y_true=event, y_pred=predict))

# %%

np.save(os.path.join(data_folder, 'cnn_predict.npy'), predict)
