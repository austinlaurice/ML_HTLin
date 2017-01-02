from math import exp
import numpy as np

def sigmoid(s):
    return float(exp(s))/(1 + exp(s))

def grad_err(w, x, y)
    return sigmoid(-1*y*np.dot(w, x))*(-1)*y*x

def gradient(w, X, Y):
    N = X.shape[0]
    grad = np.zeros(X.shape[1])
    for x, i in zip(X, range(X.shape[0])):
        grad += grad_err(w, x, Y[i])
    grad = grad/N
    return grad

def Fixed_rate_Logistic_regression_GD(X, Y, eta, T):
    w = np.zeros(X.shape[1])
    for i in range(T):
        grad = gradient(w, X, Y)
        w = w - eta * grad
    return w

def Fixed_rate_Stochastic_Logistic_regression_GD(X, Y, eta, T):
    n = 0
    N = X.shape[0]
    w = np.zeros(X.shape[1])
    for i in T:
        n = n % N
        w = w - eta * grad_err(x, X[n], Y[n])
        n += 1
    return w

def 0_1_error_test(w, X, Y):
    correct = 0
    for i in range(X.shape[0]):
        if np.dot(w, x) == Y[i]:
            correct += 1
    correct_rate = float(correct)/X.shape[0]
    return correct_rate

    

