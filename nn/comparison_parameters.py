import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def parity(k,d,n):
    assert d >= k
    X = ((((torch.rand(n,d) < 0.5)) * 2 - 1).float()).to(device)
    Y = torch.prod(X[:,:k],dim=1)
    return X,Y.unsqueeze(dim=1)

def staircase(k,l,d,n):
    assert d >= k
    assert k >= l and l > 0
    X = ((((torch.rand(n,d) < 0.5)) * 2 - 1).float()).to(device)
    # Y = torch.prod(X[:,:k],dim=1)
    Y = torch.zeros([n]).to(device)
    for i in range(k-l+1):
        Y += torch.prod(X[:,:k-i],dim=1)
    Y = torch.sign(Y)
    Y[Y==0] = -1
    return X,Y.unsqueeze(dim=1)

k = 2 # degree
l = 3 # leap
d = 2
b_size = 50

num = 5

# model parameters
M = 7000
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

# 重みの初期値
torch.nn.init.normal_(mfld.fc1.weight, mean=0, std=1)
torch.nn.init.normal_(mfld.fc2.weight, mean=0, std=1)
torch.nn.init.normal_(ntk.fc1.weight, mean=0, std=1)
torch.nn.init.normal_(ntk.fc2.weight, mean=0, std=1)

mfld_params = []
ntk_params = []

# 最初のパラメータの保存
for p in mfld.parameters():
    xs = []
    ys = []
    for i, single_p in enumerate(p.data):
        if i == num:
            break
        xs.append(single_p[0].item())
        ys.append(single_p[1].item())
    mfld_params.append({'x': xs, 'y': ys})
    break

for p in ntk.parameters():
    xs = []
    ys = []
    for i, single_p in enumerate(p.data):
        if i == num:
            break
        xs.append(single_p[0].item())
        ys.append(single_p[1].item())
    ntk_params.append({'x': xs, 'y': ys})
    break



skip = 10
for t in tqdm(range(T)):
    # X, Y = datasets(b_size, d)
    X,Y = parity(k,d,b_size)
    # X,Y = staircase(k,l,d,b_size)

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
    
            xs = []
            ys = []
            for i, single_p in enumerate(p.data):
                if i == num:
                    break
                xs.append(single_p[0].item())
                ys.append(single_p[1].item())
            mfld_params.append({'x': xs, 'y': ys})
            break

        for p in ntk.parameters():
    
            xs = []
            ys = []
            for i, single_p in enumerate(p.data):
                if i == num:
                    break
                xs.append(single_p[0].item())
                ys.append(single_p[1].item())
            ntk_params.append({'x': xs, 'y': ys})
            break

#テスト精度の算出
mfld.eval()
ntk.eval()
Xt, Yt = X,Y = parity(k,d,b_size)
mfld_loss = criterion(mfld.forward(Xt), Yt)
ntk_loss = criterion(ntk.forward(Xt), Yt)
print('mfld_loss={}'.format(mfld_loss))
print('ntk_loss={}'.format(ntk_loss))


map = plt.get_cmap('plasma')
total = T / skip

# mfldの描画
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
for i, mfld_param in enumerate(mfld_params):
    if i == 0:
        plt.scatter(mfld_param['x'], mfld_param['y'], color=map(i/total), s=100, label="initial")
    elif i == len(mfld_params)-1:
        plt.scatter(mfld_param['x'], mfld_param['y'], color=map(i/total), s=100, label="trained")
    else:
        plt.scatter(mfld_param['x'], mfld_param['y'], color=map(i/total), s=30)

plt.legend()
plt.savefig('/workspace/nn/results/mfld_parameters.png')

# ntkの描画
plt.clf()
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
for i, ntk_param in enumerate(ntk_params):
    if i == 0:
        plt.scatter(ntk_param['x'], ntk_param['y'], color=map(i/total), s=100, label="initial")
    elif i == len(ntk_params)-1:
        plt.scatter(ntk_param['x'], ntk_param['y'], color=map(i/total), s=100, label="trained")
    else:
        plt.scatter(ntk_param['x'], ntk_param['y'], color=map(i/total), s=30)

plt.legend()
plt.savefig('/workspace/nn/results/ntk_parameters.png')