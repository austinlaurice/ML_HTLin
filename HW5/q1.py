from cvxopt import matrix
from cvxopt.solvers import qp
import numpy
from svmutil import *
import SVM

def question1(y):
    N = 7
    #G = matrix([[1.0, 0.0, 0.0], [1.0, -3.0, -2.0], [1.0, 3.0, -5.0], [1.0, 3.0, -1.0], [-1.0, -5.0, 2.0], [-1.0, -9.0, 7.0], [-1.0, -9.0, -1.0], [-1.0, -9.0, -1.0]])
    #print G.size
    G = matrix([[ 1.0 , 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], [-3.0, 3.0, 3.0, -5.0, -9.0, -9.0, -9.0], [-2.0, -5.0, -1.0, 2.0, 7.0, -1.0, -1.0]])
    print G.size
    q = matrix([0.0, 0.0, 0.0])
    P = matrix([[0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])
    h = matrix([ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    sol = qp(P, q, G, h)
    print 'Question1:'
    print sol['x'], sol['y']

def Kernel(x1, x2):
    return (2 + numpy.dot(x1, x2))^2


def question2(y):
    X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
    clf = SVM.poly_kernel(y, X, 1, 2, 2, 1, 1000000)
    print SVM.get_dual_coef(clf)
    print SVM.get_SV(clf)
'''
def question2(y):
    X = numpy.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    G = matrix([[1, 0, 0, -1, 0, 0, -2], [0, 1, -1, 0, 2, -2, 0]])
    p = matrix([ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    h = matrix([ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    Q = np.zeros((7, 7), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            Q[i][j] = y[i]*y[j]*Kernel(X[i], X[j])
    
    
    h = matrix([ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
'''

if __name__ == '__main__':
    y = [-1, -1, -1, 1, 1, 1, 1]
    #X = [[1, -3, -2], [1, 3, -5], [1, 3, -1], [1, 5, -2], [1, 9, -7], [1, 9, 1], [1, 9, 1]]
    question1(y)
    question2(y)
