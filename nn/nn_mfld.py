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
    train_dataset, test_dataset, train_dataloader, test_dataloader = sample_data.get_mnist_dataloader()
    image_size = 28*28
elif dataset == 'usps':
    train_dataset, test_dataset, train_dataloader, test_dataloader = sample_data.get_usps_dataloader()
    image_size = 16*16
elif dataset == 'covtype':
    sample_data.get_covtype_dataloader()

M = 3000
lr = 1
lda1 = 1e-5 # λ'
lda2 = 1e-5  # λ

train_logname = '/workspace/nn/results/{}/{}/train_log.csv'.format(dataset, n_epochs)
test_logname = '/workspace/nn/results/{}/{}/test_log.csv'.format(dataset, n_epochs)

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
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)/M


#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net(image_size, 10).cuda()

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss()

#----------------------------------------------------------
# 学習
def train(epoch):
    model.train()  # モデルを訓練モードにする

    loss_sum = 0
    correct = 0

    for inputs, labels in train_dataloader:
        
        # GPUが使えるならGPUにデータを送る
        inputs = inputs.cuda()
        labels = labels.cuda()

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び替える
        outputs = model.forward(inputs)

        # 損失計算
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 正解の値を取得
        pred = outputs.argmax(1)
        #正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # 勾配計算
        loss.backward()

        # 重みの更新
        for p in model.parameters():
            noise = torch.normal(mean=torch.zeros_like(p.data), std=torch.ones_like(p.data)).cuda()
            p.data = (1 - 2 * lr*M * lda1) * p.data - lr*M * p.grad + np.sqrt(2*lr*M*lda2) * noise
    
    return loss_sum.item(), correct/len(train_dataset)


#----------------------------------------------------------
# 評価
def test(epoch):
    model.eval() # モデルを評価モードにする

    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            # GPUが使えるならGPUにデータを送る
            inputs = inputs.cuda()
            labels = labels.cuda()

            # ニューラルネットワークの処理を行う
            inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
            outputs = model(inputs)

            # 損失(出力とラベルとの誤差)の計算
            loss_sum += criterion(outputs, labels)

            # 正解の値を取得
            pred = outputs.argmax(1)
            #正解数をカウント
            correct += pred.eq(labels.view_as(pred)).sum().item()

    return loss_sum.item(), correct/len(test_dataset)



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


plot_from_csv.plot(dataset, n_epochs)

