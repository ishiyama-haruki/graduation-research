import numpy as np
import matplotlib.pyplot as plt
import csv

dataset_names = ['mnist', 'letter', 'covtype', 'ijcnn1', 'usps']
# dataset_names = ['mnist', 'covtype', 'ijcnn1', 'usps']


# MFLD
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(14,8))

for dataset_name in dataset_names:
    acc_logname = '/workspace/nn/results/mfld/{}/acc.csv'.format(dataset_name)

    epoch_list = []
    acc_list = []
    with open(acc_logname) as f:
        lines = f.read().split()
        for i, line in enumerate(lines):
            line = line.split(',')
            epoch_list.append(float(line[0]))
            acc_list.append(float(line[2]))
            if i == 7:
                break

    epoch_list, acc_list = zip(*sorted(zip(epoch_list, acc_list)))
    plt.plot(epoch_list, acc_list, label=dataset_name, linewidth=3)

plt.grid(which = "major", axis = "y", color = "black", alpha = 0.8,
        linestyle = "--", linewidth = 1)
plt.legend(loc='lower left')
# ylim = plt.ylim()
plt.ylim([0.7, 1])

if ('letter' in dataset_names):
    plt.savefig('/workspace/nn/results/mfld/acc.png')
else:
    plt.savefig('/workspace/nn/results/mfld/acc_noletter.png')

# NTK
plt.figure(0)
FONT_SIZE = 25.5
plt.rc('font',size=FONT_SIZE)
fig, (ax1) = plt.subplots(1,figsize=(14,8))

#縦軸をmfldに合わせる
# plt.ylim(ylim)
plt.ylim([0.7, 1])

for dataset_name in dataset_names:
    acc_logname = '/workspace/nn/results/ntk/{}/acc.csv'.format(dataset_name)

    epoch_list = []
    acc_list = []
    with open(acc_logname) as f:
        lines = f.read().split()
        for i, line in enumerate(lines):
            line = line.split(',')
            epoch_list.append(float(line[0]))
            acc_list.append(float(line[2]))
            if i == 7:
                break

    epoch_list, acc_list = zip(*sorted(zip(epoch_list, acc_list)))
    plt.plot(epoch_list, acc_list, label=dataset_name, linewidth=3)

plt.grid(which = "major", axis = "y", color = "black", alpha = 0.8,
        linestyle = "--", linewidth = 1)
plt.legend(loc='lower left')
ylim = plt.ylim()

if ('letter' in dataset_names):
    plt.savefig('/workspace/nn/results/ntk/acc.png')
else:
    plt.savefig('/workspace/nn/results/ntk/acc_noletter.png')