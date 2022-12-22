import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scripts import sample_data
from scripts import plot_from_csv
import matplotlib.pyplot as plt
import csv
import sys

# python nn/nn_mfld_visualize.py {dataset} {relu / sigmoid } {he_normal / he_uniform / normal}
dataset = sys.argv[1]
activation_function = sys.argv[2]
initialize = sys.argv[3]

if dataset == 'mnist':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_mnist()
    image_size = 719
    output_size = 10 #仮
elif dataset == 'usps':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_usps()
    image_size = 256
    output_size = 10 #仮
elif dataset == 'covtype':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_covtype()
    image_size = 54
    output_size = 10 #仮
elif dataset == 'ijcnn1':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_ijcnn1()
    image_size = 22
    output_size = 10 #仮
elif dataset == 'letter':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_letter()
    image_size = 16
    output_size = 10 #仮
elif dataset == 'cifar10':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_cifar10()
    image_size = 3072
    output_size = 10 #仮
elif dataset == 'dna':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_dna()
    image_size = 180
    output_size = 10 #仮
elif dataset == 'aloi':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_aloi()
    image_size = 128
    output_size = 10 #仮
elif dataset == 'sector':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_sector()
    image_size = 55197
    output_size = 10 #仮
elif dataset == 'shuttle':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_shuttle()
    image_size = 9
    output_size = 10 #仮
elif dataset == 'susy':
    X, Y, Xt, Yt, train_dataset, train_dataloader, test_dataset, test_dataloader = sample_data.get_susy()
    image_size = 1
    output_size = 10 #仮

M = 5000


# # GPU(CUDA)が使えるかどうか？
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

        if activation_function == 'relu':
            x = torch.relu(x)
        elif activation_function == 'sigmoid':
            x = torch.sigmoid(x)
        return x

def visualize_w():
    model.train()  # モデルを訓練モードにする

    for inputs, labels in train_dataloader:
        # 勾配の初期化
        model.zero_grad()

        # GPUが使えるならGPUにデータを送る
        inputs = inputs[0]

        # ニューラルネットワークの処理を行う
        outputs = model.forward(inputs)

        # ヒストグラムを描画
        plt.title("w-layer")
        plt.hist(outputs.detach().flatten(), 30, range=(0,1))
        plt.savefig('/workspace/nn/results/w/{}/{}.png'.format(dataset, initialize))

        break
    return
    
#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net(image_size, output_size)

# # 重みの初期値
if initialize == 'he_normal':
    torch.nn.init.kaiming_normal_(model.fc1.weight)
    torch.nn.init.kaiming_normal_(model.fc2.weight)
elif initialize == 'he_uniform':
    torch.nn.init.kaiming_uniform_(model.fc1.weight)
    torch.nn.init.kaiming_uniform_(model.fc2.weight)

visualize_w()