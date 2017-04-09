import numpy
from matplotlib import pyplot as plt

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
