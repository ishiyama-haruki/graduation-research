import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parity(k,d,n):
    assert d >= k
    X = ((((torch.rand(n,d) < 0.5)) * 2 - 1).float()).to(device)
    Y = torch.prod(X[:,:k],dim=1)
    return X,Y.unsqueeze(dim=1)

k = 4 # degree
d = 100
b_size = 50

# model parameters
M = 2048
eta = 1e-2
T = 10000
# train_2nd = False
# bias_unit = True

# regularization
lda1 = 1e-3 # weight decay
lda2 = 1e-3 # entropy

class Mfld(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mfld, self).__init__()

        self.fc1 = nn.Linear(input_size, M)
        self.fc2 = nn.Linear(M, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
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
        x = torch.relu(x)
        x = self.fc2(x)
        x = x / np.sqrt(M)
        return x


mfld = Mfld(d, b_size).cuda()
ntk = Ntk(d, b_size).cuda()
criterion = nn.MSELoss()

skip = 10
result = np.zeros([T//skip,2])

for t in range(T):
    X,Y = parity(k,d,b_size)

    mfld_loss = criterion(Y, mfld.forward(X))
    ntk_loss = criterion(Y, ntk.forward(X))

    mfld_loss.backward()
    ntk_loss.backward()

    for p in mfld.parameters():
        noise = torch.normal(mean=torch.zeros_like(p.data), std=torch.ones_like(p.data)).cuda()
        p.data = (1 - 2 * lr * lda1) * p.data - lr * M * p.grad + np.sqrt(2*lr*lda2) * noise

    for p in ntk.parameters():
        noise = torch.normal(mean=torch.zeros_like(p.data), std=torch.ones_like(p.data)).cuda()
        p.data = (1 - 2 * lr * lda1) * p.data - lr * M * p.grad + np.sqrt(2*lr*lda2) * noise

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
plt.savefig('/workspace/nn/results/k-parity-loss.png')
