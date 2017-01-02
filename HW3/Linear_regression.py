import numpy as np

def Ridge_Regression(X, Y, Lambda):
    X_trans = np.transpose(X)
    I = np.identity(X.shape[1])
    w = np.dot(np.linalg.inv((np.dot(X_trans, X) + Lambda*I)), Y)
    return w
    #https://www.quora.com/What-is-ridge-regression-How-do-you-find-its-closed-form-solution
