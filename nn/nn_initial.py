import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scripts import sample_data
import matplotlib.pyplot as plt

X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_mnist()
image_size = 719  
output_size = 10
n_epochs = 80
M = 100
lr = 0.1

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, M)
        self.fc2 = nn.Linear(M, M)
        self.fc3 = nn.Linear(M, M)
        self.fc4 = nn.Linear(M, M)
        self.fc5 = nn.Linear(M, output_size)

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def train(epoch):
    model.train()  # モデルを訓練モードにする

    loss_sum = 0
    correct = 0
    data_num = 0

    for inputs, labels in train_dataloader:
        # 勾配の初期化
        model.zero_grad()

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.cuda()
        labels = labels.cuda()

        # ニューラルネットワークの処理を行う
        outputs = model.forward(inputs)

        # 損失計算
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 正解の値を取得
        pred = outputs.argmax(1)
        #正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

        #データ数をカウント
        data_num += len(inputs)

        # 勾配計算
        loss.backward()

        # 重みの更新
        for p in model.parameters():
            p.data -= lr * p.grad
    
    return loss_sum.item()/data_num, correct/data_num

def test(epoch):
    model.eval() # モデルを評価モードにする

    loss_sum = 0
    correct = 0
    data_num = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # GPUが使えるならGPUにデータを送る
            inputs = inputs.cuda()
            labels = labels.cuda()

            # ニューラルネットワークの処理を行う
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            labels = labels.long()
            loss_sum += criterion(outputs, labels)

            # 正解の値を取得
            pred = outputs.argmax(1)
            #正解数をカウント
            correct += pred.eq(labels.view_as(pred)).sum().item()

            #データ数をカウント
            data_num += len(inputs)

    return loss_sum.item()/data_num, correct/data_num

criterion = nn.CrossEntropyLoss()

initials = {'std=0.01': [], 'Xavier': [], 'He': []}
epoch_list = []

for initial in initials.keys():
    print('{}---------------'.format(initial))
    loss_list = []
    model = Net(image_size, output_size).cuda()
    if (initial == 'std=0.01'):
        torch.nn.init.normal_(model.fc1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(model.fc2.weight, mean=0, std=0.01)
        torch.nn.init.normal_(model.fc3.weight, mean=0, std=0.01)
        torch.nn.init.normal_(model.fc4.weight, mean=0, std=0.01)
        torch.nn.init.normal_(model.fc5.weight, mean=0, std=0.01)
    elif (initial == 'Xavier'):
        torch.nn.init.xavier_normal_(model.fc1.weight, gain=1.0)
        torch.nn.init.xavier_normal_(model.fc2.weight, gain=1.0)
        torch.nn.init.xavier_normal_(model.fc3.weight, gain=1.0)
        torch.nn.init.xavier_normal_(model.fc4.weight, gain=1.0)
        torch.nn.init.xavier_normal_(model.fc5.weight, gain=1.0)
    else:
        torch.nn.init.kaiming_normal_(model.fc1.weight)
        torch.nn.init.kaiming_normal_(model.fc2.weight)
        torch.nn.init.kaiming_normal_(model.fc3.weight)
        torch.nn.init.kaiming_normal_(model.fc4.weight)
        torch.nn.init.kaiming_normal_(model.fc5.weight)


    for epoch in range(n_epochs):
        train_loss, train_acc = train(epoch)
        loss_list.append(train_loss)
        if (initial=='He'):
            epoch_list.append(epoch+1)
        print("epoch", epoch+1, " train_loss:{:.5f}".format(train_loss), "train_acc:{:.2f}".format(train_acc))
        
        # if (epoch+1)%5 == 0:
        #     test_loss, test_acc = test(epoch)
        #     loss_list.append(test_loss)
        #     if (initial=='he'):
        #         epoch_list.append(epoch+1)
        #     print("test_loss:{:.3f}".format(test_loss), "test_acc:{:.3f}".format(test_acc))

    initials[initial] = loss_list

plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(12,8))

for initial, loss_list in initials.items():
    plt.plot(epoch_list, loss_list,linewidth=3,label=initial)

plt.legend()
plt.xlabel('epochs')
plt.savefig('/workspace/nn/results/initial.png')