import numpy as np
import plotly.plotly as py

def process_data(filename):
    x_arr = []
    y_arr = []
    with open(filename, 'r') as f:
        for line in f:
            tmp = [1, ]
            line = line.strip('\n').replace('\t', ' ').split(' ')
            for x in line[:-1]:
                tmp.append(float(x))
            x_arr.append(tmp)
            y_arr.append(int(line[-1]))
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return x_arr, y_arr

def histogram(x, y, filename):
    fig = {'data': [{'x': x, 'y': y, 'type':'bar'}]}
    py.image.save_as(fig, 'filename')
