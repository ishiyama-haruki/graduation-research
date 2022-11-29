import sys
import matplotlib.pyplot as plt
import sample_data

dataset = sys.argv[1]

if dataset == 'mnist':
    train_dataset, test_dataset, train_dataloader, test_dataloader = sample_data.get_mnist_dataloader()
elif dataset == 'usps':
    train_dataset, test_dataset, train_dataloader, test_dataloader = sample_data.get_usps_dataloader()


# MNISTデータの表示
W = 5  # 横に並べる個数
H = 5   # 縦に並べる個数
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
for i in range(W*H):
    ax = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax.imshow(train_dataset[i][0][0], cmap='gray')

plt.savefig('/workspace/nn/results/{}/sample.png'.format(dataset))