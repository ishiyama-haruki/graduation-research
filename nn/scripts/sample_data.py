import torch
from torchvision import datasets, transforms
from sklearn.datasets import fetch_covtype
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


def get_covtype_dataloader():
    print('get_covtype_dataloader')
    forest = fetch_covtype()
    Data= forest['data']
    label = forest['target']
    print(Data.shape)
    print('---------------------')
    print(len(label))