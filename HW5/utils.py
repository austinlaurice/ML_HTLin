import numpy
from matplotlib import pyplot as plt
import random

def load_data(filename):
    label = []
    data = []
    with open(filename, 'r') as f:
        for line_space in f:
            line = []
            for x in line_space.split(' '):
                if x != '':
                    line.append(float(x))
            label.append(line[0])
            data.append(line[1:])
        label = numpy.array(label)
        data = numpy.array(data)
        return label, data

def split_data(label, data, sample_num):
    sampling = random.sample(range(1, len(label)), 1000)
    label_test = numpy.array([label[i] for i in sampling])
    data_test = numpy.array([data[i] for i in sampling])
    label_train = list(label[:])
    data_train = list(data[:])
    for i in sorted(sampling, reverse=True):
        del(label_train[i])
        del(data_train[i])
    return label_test, data_test, numpy.array(label_train), numpy.array(data_train)


def which_binary(label, target):
    label_new = []
    for i in range(len(label)):
        if label[i] == target:
            label_new.append(1)
        else:
            label_new.append(-1)
    label_new = numpy.array(label_new)
    return label_new


def curve(x, y, filename, xlabel, ylabel):
    plt.plot(x, y, 'r--', x, y, 'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()

def histogram(x, y, filename, xlabel, ylabel):
    plt.bar(x, y, align='center')
    fig, ax = plt.subplots()
    ax.bar(x, y, align='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(filename)
    plt.clf()
