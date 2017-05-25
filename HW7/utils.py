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


def curve(x, y, filename, xlabel, ylabel):
    plt.plot(x, y, 'r--', x, y, 'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()
