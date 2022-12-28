import numpy as np
import matplotlib.pyplot as plt
import csv

def plot(dataset, activation_function, n_epochs, ntk=False):

    if ntk:
        train_logname = '/workspace/nn/results/ntk/{}/{}/{}/train_log.csv'.format(dataset, activation_function, n_epochs)
        test_logname = '/workspace/nn/results/ntk/{}/{}/{}/test_log.csv'.format(dataset, activation_function, n_epochs)
    else:
        train_logname = '/workspace/nn/results/mfld/{}/{}/{}/train_log.csv'.format(dataset, activation_function, n_epochs)
        test_logname = '/workspace/nn/results/mfld/{}/{}/{}/test_log.csv'.format(dataset, activation_function, n_epochs)

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

    if ntk:
        plt.savefig('/workspace/nn/results/ntk/{}/{}/{}/accuracy.png'.format(dataset, activation_function, n_epochs))
    else:
        plt.savefig('/workspace/nn/results/mfld/{}/{}/{}/accuracy.png'.format(dataset, activation_function, n_epochs))