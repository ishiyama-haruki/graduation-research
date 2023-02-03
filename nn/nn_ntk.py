import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scripts import sample_data
from scripts import plot_from_csv
import csv
import sys

dataset = sys.argv[1]
n_epochs = int(sys.argv[2])

if dataset == 'mnist':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_mnist()
elif dataset == 'usps':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_usps()
elif dataset == 'covtype':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_covtype()
elif dataset == 'ijcnn1':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_ijcnn1()
elif dataset == 'letter':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_letter()
elif dataset == 'cifar10':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_cifar10()
elif dataset == 'dna':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_dna()
elif dataset == 'aloi':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_aloi()
elif dataset == 'sector':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_sector()
elif dataset == 'shuttle':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_shuttle()
elif dataset == 'susy':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_susy()

if dataset == 'mnist':
    M = 7000
    lr = 1
    lda1 = 1e-7 # λ'  l2正則化項
    image_size = 719  
    output_size = 10
elif dataset == 'usps':
    M = 7000
    lr = 1
    lda1 = 1e-7 # λ'  l2正則化項
    image_size = 16*16
    output_size = 10
elif dataset == 'covtype':
    M = 7000
    lr = 1
    lda1 = 1e-7 # λ'  l2正則化項
    image_size = 54
    output_size = 7
elif dataset == 'ijcnn1':
    M = 7000
    lr = 1
    lda1 = 1e-7 # λ'  l2正則化項
    image_size = 22
    output_size = 2
elif dataset == 'letter':
    M = 7000
    lr = 1
    lda1 = 1e-7 # λ'  l2正則化項
    image_size = 16
    output_size = 26


train_logname = '/workspace/nn/results/ntk/{}/{}/train_log.csv'.format(dataset, n_epochs)
test_logname = '/workspace/nn/results/ntk/{}/{}/test_log.csv'.format(dataset, n_epochs)

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#----------------------------------------------------------
# ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, M)
        self.fc2 = nn.Linear(M, output_size)

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x / np.sqrt(M)
        return x


#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net(image_size, output_size).cuda()

# # 重みの初期値
torch.nn.init.normal_(model.fc1.weight, mean=0, std=1)
torch.nn.init.normal_(model.fc2.weight, mean=0, std=1)

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss()

#----------------------------------------------------------
# 学習
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
            p.data -= lr * (p.grad + 2*lda1*p.data)  
    
    return loss_sum.item()/data_num, correct/data_num


#----------------------------------------------------------
# 評価
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



for epoch in range(n_epochs):
    train_loss, train_acc = train(epoch)
    print("epoch", epoch+1, " train_loss:{:.5f}".format(train_loss), "train_acc:{:.2f}".format(train_acc))
    
    with open(train_logname, 'a') as train_logfile:
        train_logwriter = csv.writer(train_logfile, delimiter=',')
        train_logwriter.writerow([epoch, "{:.5f}".format(float(train_loss)), "{:.3f}".format(float(train_acc))])
    
    if (epoch+1)%5 == 0:
        test_loss, test_acc = test(epoch)
        print("test_loss:{:.3f}".format(test_loss), "test_acc:{:.3f}".format(test_acc))
        
        with open(test_logname, 'a') as test_logfile:
            test_logwriter = csv.writer(test_logfile, delimiter=',')
            test_logwriter.writerow([epoch, "{:.5f}".format(float(test_loss)), "{:.3f}".format(float(test_acc))])

plot_from_csv.plot(dataset, n_epochs, 'ntk')

