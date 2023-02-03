import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scripts import sample_data
import matplotlib.pyplot as plt
import csv
import sys

n_epochs = int(sys.argv[1])

dataloader_dict = {
    'mnist': {
        'train': sample_data.get_mnist()[5],
        'test': sample_data.get_mnist()[7],
    },
    'letter': {
        'train': sample_data.get_letter()[5],
        'test': sample_data.get_letter()[7],
    },
    'usps': {
        'train': sample_data.get_usps()[5],
        'test': sample_data.get_usps()[7],
    },
    'covtype': {
        'train': sample_data.get_covtype()[5],
        'test': sample_data.get_covtype()[7],
    },
    'ijcnn1': {
        'train': sample_data.get_ijcnn1()[5],
        'test': sample_data.get_ijcnn1()[7],
    }
}

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
        x = x / M
        return x


#----------------------------------------------------------
# 学習
def train(dataloader):
    model.train()  # モデルを訓練モードにする

    loss_sum = 0
    correct = 0
    data_num = 0

    for inputs, labels in dataloader:
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
            noise = torch.normal(mean=torch.zeros_like(p.data), std=torch.ones_like(p.data)).cuda()
            p.data -= lr * (M * p.grad + 2*lda1*p.data) + np.sqrt(2*lr*lda2) * noise
    
    return loss_sum.item()/data_num, correct/data_num

#----------------------------------------------------------
# 評価
def test(test_dataloader):
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


result_dict = {}
for dataset_name, dataloaders in dataloader_dict.items():
    print('{} start'.format(dataset_name))
    # λ' L2正則加項
    # λ  mfld
    if dataset_name == 'mnist':
        M = 5000
        lr = 1
        lda1 = 1e-7
        lda2 = 1e-5
        image_size = 719  
        output_size = 10
    elif dataset_name == 'usps':
        M = 1000
        lr = 1
        lda1 = 1e-5
        lda2 = 1e-7 
        image_size = 16*16
        output_size = 10
    elif dataset_name == 'covtype':
        M = 7000
        lr = 1
        lda1 = 1e-7
        lda2 = 1e-5 
        image_size = 54
        output_size = 7
    elif dataset_name == 'ijcnn1':
        M = 5000
        lr = 1
        lda1 = 1e-7
        lda2 = 1e-5 
        image_size = 22
        output_size = 2
    elif dataset_name == 'letter':
        M = 7000
        lr = 1
        lda1 = 1e-7
        lda2 = 1e-5 
        image_size = 16
        output_size = 26

    model = Net(image_size, output_size).cuda()
    # 重みの初期値
    torch.nn.init.normal_(model.fc1.weight, mean=0, std=1)
    torch.nn.init.normal_(model.fc2.weight, mean=0, std=1)
    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        train_loss, train_acc = train(dataloaders['train'])

        if (epoch+1)%5 == 0:
            test_loss, test_acc = test(dataloaders['test'])

            with open('/workspace/nn/results/mfld/{}/loss.csv'.format(dataset_name), 'a') as test_logfile:
                test_logwriter = csv.writer(test_logfile, delimiter=',')
                test_logwriter.writerow([epoch, "{:.5f}".format(float(test_loss)), "{:.3f}".format(float(test_acc))])
    
    
    