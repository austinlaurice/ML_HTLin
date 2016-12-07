import numpy as np
import plotly.plotly as py

def process_data(filename):
    x_arr = []
    y_arr = []
    with open(filename, 'r') as f:
        for line in f:
            tmp = [1]
            x_arr.append(tmp.extend([int(x) for x in line.split(' ')[:-1]]))
            y_arr.append(int(line.split(' ')[-1]))
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return x_arr, y_arr

def histogram(x, y, filename):
    fig = {'data': [{'x': x, 'y': y, 'type'= 'bar'}]}
    py.image.save_as(fig, 'filename')
