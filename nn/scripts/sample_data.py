import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

n_batch = 100

def get_mnist_dataloader():

    # 学習用
    train_dataset = datasets.MNIST(
        '/workspace/nn/data',               # データの保存先
        train = True,           # 学習用データを取得する
        download = True,        # データが無い時にダウンロードする
        transform = transform   # テンソルへの変換など
        )

    save_data_img(train_dataset, 'mnist')

    # 評価用
    test_dataset = datasets.MNIST(
        '/workspace/nn/data', 
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

    return train_dataset, test_dataset, train_dataloader, test_dataloader


def get_usps_dataloader():

    # 学習用
    train_dataset = datasets.USPS(
        '/workspace/nn/data',               # データの保存先
        train = True,           # 学習用データを取得する
        download = True,        # データが無い時にダウンロードする
        transform = transform   # テンソルへの変換など
        )

    save_data_img(train_dataset, 'usps')

    # 評価用
    test_dataset = datasets.USPS(
        '/workspace/nn/data', 
        train = False,
        download = True,   
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

    return train_dataset, test_dataset, train_dataloader, test_dataloader



def save_data_img(train_dataset, dataset_name):
    # MNISTデータの表示
    W = 5  # 横に並べる個数
    H = 5   # 縦に並べる個数
    fig = plt.figure(figsize=(H, W))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
    for i in range(W*H):
        ax = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
        ax.imshow(train_dataset[i][0][0], cmap='gray')

    plt.savefig('/workspace/nn/results/{}/sample.png'.format(dataset_name))