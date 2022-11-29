import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scripts import sample_data
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader
import sys
import time
import itertools

dataset = sys.argv[1]
n_epochs = int(sys.argv[2])


if dataset == 'mnist':
    learning_dataset, test_dataset, learning_dataloader, test_dataloader = sample_data.get_mnist_dataloader()
    image_size = 28*28
elif dataset == 'usps':
    learning_dataset, test_dataset, learning_dataloader, test_dataloader = sample_data.get_usps_dataloader()
    image_size = 16*16

n_batch = 128
M = [1000, 3000, 5000]
lr = [1, 1e-1, 1e-2]
lda1 = [1e-3, 1e-5, 1e-7] # λ'
lda2 = [1e-3, 1e-5, 1e-7]  # λ
params = list(itertools.product(M, lr, lda1, lda2))

print('nn_mfld_kfold_gs.py start!')
sys.stdout.flush() # 明示的にflush


# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------
# ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        print('M: {}'.format(M))
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
# 学習
def train(train_dataset, train_dataloader, M, lr, lda1, lda2):
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
def test(dataset, dataloader):
    model.eval() # モデルを評価モードにする

    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
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

    return loss_sum.item(), correct/len(dataset)


max_val_acc = 0
best_param = []
# グリッドサーチ：全パラメータの組み合わせで実行
for i, param in enumerate(params): 
    print('{}/{}'.format(i+1, len(params)))
    print(param)
    start_time = time.time()

    M = param[0]
    lr =  param[1]
    lda1 = param[2]
    lda2 = param[3]

    # ニューラルネットワークの生成
    model = Net(image_size, 10).cuda()
    #----------------------------------------------------------
    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # クロスバリデーション
    kf = KFold(n_splits=3, shuffle=True)
    val_acc_sum = 0
    for train_index, valid_index in kf.split(learning_dataset.data):
        train_dataset = Subset(learning_dataset, train_index)
        train_dataloader = DataLoader(train_dataset, n_batch, shuffle=True)
        valid_dataset   = Subset(learning_dataset, valid_index)
        valid_dataloader = DataLoader(valid_dataset, n_batch, shuffle=False)

        for epoch in range(n_epochs):
            train_loss, train_acc = train(train_dataset, train_dataloader, M, lr, lda1, lda2)
            # print('epoch: {} train_loss: {} train_acc: {}'.format(epoch, train_loss, train_acc))
        val_loss, val_acc = test(valid_dataset, valid_dataloader)
        val_acc_sum += val_acc

    val_acc_mean = val_acc_sum / kf.n_splits
    print('val_acc_mean = {}'.format(val_acc_mean))

    end_time = time.time()
    print('time: {}'.format(end_time- start_time))

    print('-------------------------------------------------------------------------')

    sys.stdout.flush() # 明示的にflush

    if (val_acc_mean > max_val_acc):
        max_val_acc = val_acc_mean
        best_param = param
        torch.save(model, '/workspace/nn/results/{}/{}/model_weight.pth'.format(dataset, n_epochs))

print('best param')
print(best_param)
print('max_val_acc = {}'.format(max_val_acc))


# ベストスコアを出したモデルでテストスコアを出す
model = torch.load('/workspace/nn/results/{}/{}/model_weight.pth'.format(dataset, n_epochs))
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = test(test_dataset, test_dataloader)
print('test_acc = {}'.format(test_acc))