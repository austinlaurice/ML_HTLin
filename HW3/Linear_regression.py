import numpy as np

def Ridge_Regression(X, Y, Lambda):
    X_trans = np.transpose(X)
    I = np.identity(X.shape[1])
    #print np.dot(X_trans, X)
    w = np.dot(np.dot(np.linalg.inv((np.dot(X_trans, X) + Lambda*I)), X_trans), Y)
    return w
    #https://www.quora.com/What-is-ridge-regression-How-do-you-find-its-closed-form-solution

def find_best_lambda(E, L):
    error = 100
    ind = 0
    for i in range(len(E)):
        if E[i] <= error:
            error = E[i]
            ind = i
    return i
