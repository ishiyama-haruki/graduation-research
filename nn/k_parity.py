import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, M)
        self.fc2 = nn.Linear(M, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x / M
        return x


model = Net(d+1, b_size).cuda()
criterion = nn.MSELoss()

skip = 10
result = np.zeros([T//skip,3])


for t in tqdm(range(T)):
    X,Y = parity(k,d,b_size)

    loss = criterion(Y, model.forward(X))

    loss.backward()

    for p in model.parameters():
        noise = torch.normal(mean=torch.zeros_like(p.data), std=torch.ones_like(p.data)).cuda()
        p.data = (1 - 2 * lr * lda1) * p.data - lr * M * p.grad + np.sqrt(2*lr*lda2) * noise

    # if t % skip == 0:        

    #     result[t//skip,0] = loss_fn(Y,model.forward(X))
    #     result[t//skip,1] = zero_one(Y,model.forward(X))
    #     result[t//skip,2] = torch.norm(Ws[:k,:])**2 / torch.norm(Ws)**2

# plt.figure(0)
# FONT_SIZE = 25.5
# plt.rc('font',size=FONT_SIZE)
# fig, (ax1) = plt.subplots(1,figsize=(10,8))
