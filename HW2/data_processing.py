import numpy as np
from matplotlib import pyplot as plt

def process_data(filename):
    x_arr = []
    y_arr = []
    with open(filename, 'r') as f:
        for line in f:
            tmp = []
            line = line.strip('\n').replace('\t', ' ').split(' ')
            for x in line[:-1]:
                tmp.append(float(x))
            x_arr.append(tmp)
            y_arr.append(int(line[-1]))
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return x_arr, y_arr

# One dimension X and Y
def sort_x_y(X, Y):
    index_order = X.argsort()
    X.sort()
    Y = Y(index_order)
    return X, Y



#def histogram(x, y, filename):
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

    
