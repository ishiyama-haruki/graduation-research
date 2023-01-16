import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

k = 4 # degree
l = 3 # leap
d = 100
b_size = 50

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

        self.fc1 = nn.Linear(input_size, M, bias=False)
        self.fc2 = nn.Linear(M, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = x / M
        return x

class Ntk(nn.Module):
    def __init__(self, input_size, output_size):
        super(Ntk, self).__init__()

        self.fc1 = nn.Linear(input_size, M, bias=False)
        self.fc2 = nn.Linear(M, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = x / np.sqrt(M)
        return x


mfld = Mfld(d+1, 1).cuda()
ntk = Ntk(d+1, 1).cuda()
criterion = nn.MSELoss()

# 重みの初期値
# torch.nn.init.xavier_normal_(mfld.fc1.weight)
# torch.nn.init.xavier_normal_(mfld.fc2.weight)
# torch.nn.init.xavier_normal_(ntk.fc1.weight)
# torch.nn.init.xavier_normal_(ntk.fc2.weight)
# torch.nn.init.normal_(mfld.fc1.weight, mean=0, std=1)
# torch.nn.init.normal_(mfld.fc2.weight, mean=0, std=1)
# torch.nn.init.normal_(ntk.fc1.weight, mean=0, std=1)
# torch.nn.init.normal_(ntk.fc2.weight, mean=0, std=1)
# torch.nn.init.normal_(mfld.fc1.weight, mean=0,std=(1/d)**0.5)
# torch.nn.init.normal_(mfld.fc2.weight, mean=0,std=(1/d)**0.5)
# torch.nn.init.normal_(ntk.fc1.weight, mean=0,std=(1/d)**0.5)
# torch.nn.init.normal_(ntk.fc2.weight, mean=0,std=(1/d)**0.5)


        

skip = 10
first_laylers = np.zeros([T//skip,2])
second_laylers = np.zeros([T//skip,2])

for t in tqdm(range(T)):
    # X,Y = parity(k,d,b_size)  
    X,Y = staircase(k,l,d,b_size)
    X = torch.cat([X,torch.ones(b_size,1).to(device)],dim=1)

    mfld.zero_grad()
    ntk.zero_grad()

    mfld_loss = criterion(mfld.forward(X), Y)
    ntk_loss = criterion(ntk.forward(X), Y)

    mfld_loss.backward()
    ntk_loss.backward()

    
    for i, p in enumerate(mfld.parameters()):
        noise = torch.FloatTensor(p.data.shape).normal_().to(device)
        p.data -= lr * (M * p.grad + 2*lda1*p.data) + np.sqrt(2*lr*lda2) * noise
        if i == 0:
            mfld_1st_param = p.data[10][10]
        else:
            mfld_2nd_param = p.data[0][10]
        

    for i, p in enumerate(ntk.parameters()):
        p.data -= lr * (p.grad + 2*lda1*p.data) 
        if i == 0:
            ntk_1st_param = p.data[10][10]
        else:
            ntk_2nd_param = p.data[0][10]

    del X,Y,noise
    torch.cuda.empty_cache()

    if t % skip == 0:        
        first_laylers[t//skip,0] = mfld_1st_param
        first_laylers[t//skip,1] = ntk_1st_param
        second_laylers[t//skip,0] = mfld_2nd_param
        second_laylers[t//skip,1] = ntk_2nd_param

plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(10,10))

axes[0].plot(np.arange(1,T+1,skip),first_laylers[:,0],linewidth=3,label='mfld first layer param')
axes[0].plot(np.arange(1,T+1,skip),first_laylers[:,1],linewidth=3,label='ntk first layer param')

axes[0].legend()
# axes[0].xlabel('GD steps')
axes[0].set_xlabel('GD steps')

axes[1].plot(np.arange(1,T+1,skip),second_laylers[:,0],linewidth=3,label='mfld second layer param')
axes[1].plot(np.arange(1,T+1,skip),second_laylers[:,1],linewidth=3,label='ntk second layer param')

axes[1].legend()
# axes[1].xlabel('GD steps')
axes[1].set_xlabel('GD steps')

plt.savefig('/workspace/nn/results/k_parity_parameter.png')