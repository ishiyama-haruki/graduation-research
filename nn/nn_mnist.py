import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

n_epochs = 30
n_batch = 100
lr = 1e-5
image_size = 28*28

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# 学習用
train_dataset = datasets.MNIST(
    './data',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    transform = transform   # テンソルへの変換など
    )

# 評価用
test_dataset = datasets.MNIST(
    './data', 
    train = False,
    transform = transform
    )

# データローダー
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = n_batch,
    shuffle = True)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = n_batch,
    shuffle = True)


#----------------------------------------------------------
# ニューラルネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net(image_size, 10)#.to(device)

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# #----------------------------------------------------------
# # 最適化手法の設定
# optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 


#----------------------------------------------------------
# 学習
model.train()  # モデルを訓練モードにする

loss_hist = []

for epoch in range(n_epochs):
    loss_sum = 0

    for inputs, labels in train_dataloader:
        
        # # GPUが使えるならGPUにデータを送る
        # inputs = inputs.to(device)
        # lables = labels.to(device)

        # # optimizerを初期化
        # optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び替える
        outputs = model.forward(inputs)

        # 損失計算
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 勾配計算
        loss.backward()

        # 重みの更新
        # optimizer.step()
        for p in model.parameters():
            p.data -= lr * p.grad
    

    loss_hist.append(loss_sum.detach().numpy())
    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{n_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

    # #モデルの重みの保存
    # torch.save(model.state_dict(), 'model_weights.pth')

plt.plot(loss_hist)
plt.ylabel("cost")
plt.xlabel("iterations")
plt.savefig('/workspace/nn/results/nn_mnist.png')


#----------------------------------------------------------
# 評価
model.eval() # モデルを評価モードにする

loss_sum = 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        # # GPUが使えるならGPUにデータを送る
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        inputs = inputs.view(-1, image_size) # 画像データ部分を一次元へ並び変える
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        #正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")