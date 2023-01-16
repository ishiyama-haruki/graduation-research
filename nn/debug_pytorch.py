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
lda2 = 1e-3 # entropy

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
result = np.zeros([T//skip,2])

for t in range(T): #tqdm(range(T)):
    # X,Y = parity(k,d,b_size)  
    X,Y = staircase(k,l,d,b_size)
    X = torch.cat([X,torch.ones(b_size,1).to(device)],dim=1)

    mfld_loss = criterion(mfld.forward(X), Y)
    ntk_loss = criterion(ntk.forward(X), Y)

    mfld_loss.backward()
    ntk_loss.backward()

    print('mfld loss = {}, ntk loss = {}'.format(mfld_loss, ntk_loss))

    for p in mfld.parameters():
        noise = torch.FloatTensor(p.data.shape).normal_().to(device)
        p.data -= lr * (M * p.grad + 2*lda1*p.data) + np.sqrt(2*lr*lda2) * noise
            

    for p in ntk.parameters():
        p.data -= lr * (p.grad + 2*lda1*p.data) 

    del X,Y,noise
    torch.cuda.empty_cache()

    if t % skip == 0:        

        result[t//skip,0] = mfld_loss
        result[t//skip,1] = ntk_loss

plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(10,8))

plt.plot(np.arange(1,T+1,skip),result[:,0],linewidth=3,label='mfld loss')
plt.plot(np.arange(1,T+1,skip),result[:,1],linewidth=3,label='ntk loss')

plt.legend()
plt.xlabel('GD steps')
plt.savefig('/workspace/nn/results/debug_pytorch.png')