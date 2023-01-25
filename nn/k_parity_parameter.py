import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def datasets(b_size, d):
    X, Y = make_blobs(
        n_samples=b_size, 
        n_features=d, 
        cluster_std=1.5,
        centers=1
    )
    X = torch.from_numpy(X.astype(np.float32)).clone().cuda()
    Y = torch.from_numpy(Y.astype(np.float32)).clone().cuda()
    return X,Y.unsqueeze(dim=1)

d = 2
b_size = 50

num = 2

# model parameters
M = 2048
lr = 1e-2
T = 10000

# regularization
lda1 = 1e-3 # weight decay
lda2 = 1e-3 # 正則化項

class Mfld(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mfld, self).__init__()

        self.fc1 = nn.Linear(input_size, M)
        self.fc2 = nn.Linear(M, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = x / M
        return x

class Ntk(nn.Module):
    def __init__(self, input_size, output_size):
        super(Ntk, self).__init__()

        self.fc1 = nn.Linear(input_size, M)
        self.fc2 = nn.Linear(M, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = x / np.sqrt(M)
        return x


mfld = Mfld(d, 1).cuda()
ntk = Ntk(d, 1).cuda()
criterion = nn.MSELoss()

mfld_params = []
ntk_params = []

# 最初のパラメータの保存
for p in mfld.parameters():
    for i, single_p in enumerate(p.data):
        if i == num:
            break
        mfld_params.append({'x': [single_p[0].item()], 'y': [single_p[1].item()]})
    break

for p in ntk.parameters():
    for i, single_p in enumerate(p.data):
        if i == num:
            break
        ntk_params.append({'x': [single_p[0].item()], 'y': [single_p[1].item()]})
    break



skip = 10
for t in tqdm(range(T)):
    X, Y = datasets(b_size, d)

    mfld.zero_grad()
    ntk.zero_grad()

    mfld_loss = criterion(mfld.forward(X), Y)
    ntk_loss = criterion(ntk.forward(X), Y)

    mfld_loss.backward()
    ntk_loss.backward()

    
    for p in mfld.parameters():
        noise = torch.FloatTensor(p.data.shape).normal_().to(device)
        p.data -= lr * (M * p.grad + 2*lda1*p.data) + np.sqrt(2*lr*lda2) * noise
            

    for p in ntk.parameters():
        p.data -= lr * (p.grad + 2*lda1*p.data) 

    del X,Y,noise
    torch.cuda.empty_cache()

    if t % skip == 0:        
        for p in mfld.parameters():
            for i, single_p in enumerate(p.data):
                if i == num:
                    break
                mfld_params[i]['x'].append(single_p[0].item())
                mfld_params[i]['y'].append(single_p[1].item())
            break
        for p in ntk.parameters():
            for i, single_p in enumerate(p.data):
                if i == num:
                    break
                ntk_params[i]['x'].append(single_p[0].item())
                ntk_params[i]['y'].append(single_p[1].item())
            break

# mfldの描画
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(10,8))

for mfld_param in mfld_params:
    plt.plot(mfld_param['x'][0], mfld_param['y'][0], '.', markersize=20, color='darkviolet')
    plt.plot(mfld_param['x'][-1], mfld_param['y'][-1], '.', markersize=20, color='y')
    plt.plot(mfld_param['x'], mfld_param['y'], color='darkviolet', linewidth=3)

plt.savefig('/workspace/nn/results/mfld_parameters.png')

# ntkの描画
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(10,8))
for ntk_param in ntk_params:
    plt.plot(ntk_param['x'][0], ntk_param['y'][0], '.', markersize=20, color='darkviolet')
    plt.plot(ntk_param['x'][-1], ntk_param['y'][-1], '.', markersize=20, color='y')
    plt.plot(ntk_param['x'], ntk_param['y'], color='darkviolet', linewidth=3)

plt.savefig('/workspace/nn/results/ntk_parameters.png')