import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt
from sympy.solvers import solve

delta = 0.05
dvc = 50

def fa(x):
    global delta, dvc
    return np.sqrt(float(8)/x * np.log((4 * (2*x)**dvc)/delta))

def fb(x):
    global delta, dvc
    return np.sqrt(float(16)/x * np.log((2 * (x**dvc))/sqrt(delta)))

def fc(x):
    global delta, dvc
    return np.sqrt(2 * np.log(2 * x * x**dvc)/ x) + np.sqrt(float(2)/x * log(1/delta)) + float(1)/x

def original():
    x = np.linspace(0.001,20, 100)
    y = fa(x)
    plt.plot(x,y,'k--')
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel('N')
    plt.ylabel('generalization error bound')
    plt.title('Original VC Bound')
    plt.savefig('qustion14_original.png')
    plt.show()


def variant():
    x = np.linspace(0.001,20, 100)
    y = fb(x)
    plt.plot(x,y,'k--')
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel('N')
    plt.ylabel('generalization error bound')
    plt.title('Variant VC Bound')
    plt.savefig('qustion14_variant.png')
    plt.show()

def rademacher():
    x = np.linspace(0.001,20, 100)
    y = fc(x)
    plt.plot(x,y,'k--')
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel('N')
    plt.ylabel('generalization error bound')
    plt.title('Rademachor Penalty Bound')
    plt.savefig('qustion14_radmacher.png')
    plt.show()

def parrondo():
    global delta, dvc
    y,x = np.ogrid[0:20:100j,0:20:100j]
    plt.contour(x.ravel(), y.ravel(), np.sqrt(1/x * (2 * y + np.log(float(6) * (2 * x)**dvc / delta))) - y, [0])
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel('N')
    plt.ylabel('generalization error bound')
    plt.title('Parrondo and Van den Brock Bound')
    plt.savefig('qustion14_parrondo.png')
    plt.show()

def devroye():
    global delta, dvc
    y,x = np.ogrid[0:40:100j,0:40:100j]
    plt.contour(x.ravel(), y.ravel(), np.sqrt(1/(2*x) * (4 * y * (1 + y) + np.log(float(4) * (x**2)**dvc/delta))) - y, [0])
    plt.xlim(0,40)
    plt.ylim(0,40)
    plt.xlabel('N')
    plt.ylabel('generalization error bound')
    plt.title('Devroye Bound')
    plt.savefig('qustion14_parrondo.png')
    plt.show()

if __name__ == '__main__':
    #original()
    #variant()
    #rademacher()
    parrondo()
    devroye()

