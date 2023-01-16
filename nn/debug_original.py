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

train_2nd = True#False

Ws_mfld = torch.normal(mean=0,std=(1/d)**0.5,size=(d+1,M),device=device,requires_grad=True)
bs_mfld =  torch.cat([torch.ones(M//2, 1),-torch.ones(M//2,1)]).to(device)

Ws_ntk = torch.normal(mean=0,std=(1/d)**0.5,size=(d+1,M),device=device,requires_grad=True)
bs_ntk =  torch.cat([torch.ones(M//2, 1),-torch.ones(M//2,1)]).to(device)

if train_2nd:
    bs_mfld.requires_grad = True
    bs_ntk.requires_grad = True

class Mfld(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mfld, self).__init__()

    def forward(self, x):
        return torch.tanh(X@Ws_mfld)@bs_mfld/M

class Ntk(nn.Module):
    def __init__(self, input_size, output_size):
        super(Ntk, self).__init__()

    def forward(self, x):
        return torch.tanh(X@Ws_ntk)@bs_ntk/np.sqrt(M)


mfld = Mfld(d+1, 1).cuda()
ntk = Ntk(d+1, 1).cuda()
criterion = nn.MSELoss()


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

    with torch.no_grad():
        Bt = torch.FloatTensor(X.shape[1],M).normal_().to(device)
        Ws_mfld -= lr * (M * Ws_mfld.grad + 2*lda1*Ws_mfld) + np.sqrt(2*lr*lda2) * Bt
        Ws_ntk -= lr * (Ws_ntk.grad + 2*lda1*Ws_ntk)

        if train_2nd:
            Bt = torch.FloatTensor(M,1).normal_().to(device)
            bs_mfld -= lr * (M * bs_mfld.grad + 2*lda1*bs_mfld) + np.sqrt(2*lr*lda2) * Bt
            bs_ntk -= lr * lr * (bs_ntk.grad + 2*lda1*bs_ntk)

    Ws_mfld.grad.zero_()
    Ws_ntk.grad.zero_()
    if train_2nd:
        bs_mfld.grad.zero_()
        bs_ntk.grad.zero_()
        
    del X,Y,Bt
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
plt.savefig('/workspace/nn/results/debug_original.png')