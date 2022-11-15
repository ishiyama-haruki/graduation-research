import numpy as np
import matplotlib.pyplot as plt
import csv

train_logname = '/workspace/nn/results/train_log.csv'
test_logname = '/workspace/nn/results/test_log.csv'

#訓練データ
epoch_list = []
accuracy_list = []
with open(train_logname) as f:
    lines = f.read().split()
    for i, line in enumerate(lines):
        line = line.split(',')
        epoch_list.append(float(line[0]))
        accuracy_list.append(float(line[2]))

epoch_list, accuracy_list = zip(*sorted(zip(epoch_list, accuracy_list)))
plt.plot(epoch_list, accuracy_list, label='train', color='b')

#テストデータ
epoch_list = []
accuracy_list = []
with open(test_logname) as f:
    lines = f.read().split()
    for i, line in enumerate(lines):
        line = line.split(',')
        epoch_list.append(float(line[0]))
        accuracy_list.append(float(line[2]))

epoch_list, accuracy_list = zip(*sorted(zip(epoch_list, accuracy_list)))
plt.plot(epoch_list, accuracy_list, label='test', color='r', linestyle='dashed')

plt.legend()
plt.savefig('/workspace/nn/results/accuracy.png')