import numpy
from matplotlib import pyplot as plt
import random

def load_data(filename):
    feature1 = []
    feature2 = []
    label = []
    data = []
    c = 0
    with open(filename, 'r') as f:
        for line_space in f:
            line = []
            for x in line_space.split(' '):
                if x != '':
                    line.append(float(x))
            feature1.append((line[0], line[-1], c))
            feature2.append((line[1], line[-1], c))
            c += 1
    feature1 = sorted(feature1, key=lambda k: k[0])
    feature2 = sorted(feature2, key=lambda k: k[0])
    return [feature1, feature2]

def load_data_tree(filename):
    features = []
    with open(filename, 'r') as f:
        for line_space in f:
            line = []
            for x in line_space.split(' '):
                if x != '':
                    line.append(float(x))
            features.append(line)    
    return features

def histogram(x, filename, xlabel, ylabel):
    hist, bins = np.histogram(x, bins='auto')
    width = 0.75 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    #plt.show()
    fig, ax = plt.subplots()
    ax.bar(center, hist, align='center', width=width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(filename)

def histogram_2(x, y, filename, xlabel, ylabel):
    plt.bar(x, y, align='center')
    fig, ax = plt.subplots()
    ax.bar(x, y, align='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def curve(x, y, filename, xlabel, ylabel):
    plt.plot(x, y, 'r--', x, y, 'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()
