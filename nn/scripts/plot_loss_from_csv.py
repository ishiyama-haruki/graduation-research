import numpy as np
import matplotlib.pyplot as plt
import csv

# dataset_names = ['mnist', 'letter', 'covtype', 'ijcnn1', 'usps']
dataset_names = ['mnist', 'covtype', 'ijcnn1', 'usps']


# MFLD
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(14,8))

for dataset_name in dataset_names:
    loss_logname = '/workspace/nn/results/mfld/{}/loss.csv'.format(dataset_name)

    epoch_list = []
    loss_list = []
    with open(loss_logname) as f:
        lines = f.read().split()
        for i, line in enumerate(lines):
            line = line.split(',')
            epoch_list.append(float(line[0]))
            loss_list.append(float(line[1]))

    epoch_list, loss_list = zip(*sorted(zip(epoch_list, loss_list)))
    plt.plot(epoch_list, loss_list, label=dataset_name, linewidth=3)

plt.legend()
ylim = plt.ylim()

if ('letter' in dataset_names):
    plt.savefig('/workspace/nn/results/mfld/loss.png')
else:
    plt.savefig('/workspace/nn/results/mfld/loss_noletter.png')

# NTK
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(14,8))

#縦軸をmfldに合わせる
plt.ylim(ylim)

for dataset_name in dataset_names:
    loss_logname = '/workspace/nn/results/ntk/{}/loss.csv'.format(dataset_name)

    epoch_list = []
    loss_list = []
    with open(loss_logname) as f:
        lines = f.read().split()
        for i, line in enumerate(lines):
            line = line.split(',')
            epoch_list.append(float(line[0]))
            loss_list.append(float(line[1]))

    epoch_list, loss_list = zip(*sorted(zip(epoch_list, loss_list)))
    plt.plot(epoch_list, loss_list, label=dataset_name, linewidth=3)

plt.legend()
ylim = plt.ylim()

if ('letter' in dataset_names):
    plt.savefig('/workspace/nn/results/ntk/loss.png')
else:
    plt.savefig('/workspace/nn/results/ntk/loss_noletter.png')