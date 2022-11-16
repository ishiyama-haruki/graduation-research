import torch
from torchvision import datasets, transforms

# 変換方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
    ])

def get_mnist_dataloader():
    n_batch = 100

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
