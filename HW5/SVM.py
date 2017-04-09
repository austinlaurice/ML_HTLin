#from svmutil import *
from utils import *
from sklearn import svm
import numpy

def cal_w_dis(clf):
    w = clf.coef_
    #import ipdb; ipdb.set_trace()
    sum = 0.0
    for i in w[0]:
        sum += i**2
    return sum**0.5

def error_0_1(ans, data, clf):
    result = clf.predict(data)
    err = 0
    for i in range(len(ans)):
        if ans[i]!= result[i]:
            err += 1
    return float(err)/len(ans)

def SV_num(clf):
    return clf.n_support_

def get_SV(clf):
    return clf.support_vectors_

def get_coef(clf):
    return clf.coef_

def get_dual_coef(clf):
    return clf.dual_coef_

def free_SV(clf):
    pass

def print_info(clf):
    print clf.coef_
    print clf.n_support_
    print clf.support_vectors_

def linear_kernel(label, data, binary, C):
    print C
    label = which_binary(label, binary)
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(data, label)
    return clf
    #return cal_w_dis(clf)
    #print_info(clf)
    #print clf.coef_

def poly_kernel(label, data, binary, degree, coef, gamma, C):
    label = which_binary(label, binary)
    print numpy.unique(label)
    clf = svm.SVC(kernel='poly', degree=degree, coef0=coef, gamma=gamma, C=C)
    clf.fit(data, label)
    return clf
    #print_info(clf)

def gaussian_kernel(label, data, binary, gamma, C):
    label = which_binary(label, binary)
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
    clf.fit(data, label)
    return clf
    #print_info(clf)

