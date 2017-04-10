#from svmutil import *
from utils import *
from sklearn import svm
import numpy
import math

def cal_w_dis(clf):
    w = clf.coef_
    #import ipdb; ipdb.set_trace()
    sum_coef = 0.0
    for i in w[0]:
        sum_coef += i**2
    return sum_coef**0.5

def cal_gaussian_kernel(x1, x2, gamma):
    x = numpy.subtract(x1, x2)
    expo = (-1)*gamma*(x[0]**2 + x[1]**2)
    return math.exp(expo)

def cal_dis(SV, SV_coef, x):
    # kernel = exp(-80|||xn-xm||^2)
    sigma = 0
    for i in xrange(len(SV)):
        for j in xrange(len(SV)):
            sigma += SV_coef[i] * SV_coef[j] * cal_gaussian_kernel(SV[i], SV[j], 80)
    return 1/math.sqrt(sigma)
        


def error_0_1(ans, data, clf):
    result = clf.predict(data)
    err = 0
    for i in range(len(ans)):
        if ans[i]!= result[i]:
            err += 1
    return float(err)/len(ans)

def SV_num(clf):
    ans = sum(clf.n_support_)
    return ans

def get_SV(clf):
    return clf.support_vectors_

def get_coef(clf):
    return clf.coef_

def get_dual_coef(clf):
    return clf.dual_coef_

def free_SV(clf, C):
    SV = clf.support_vectors_
    coef = clf.dual_coef_
    free_SV = []
    free_SV_coef = []
    for i in xrange(len(coef[0])):
        if abs(coef[0][i]) < C:
            free_SV.append(SV[i])
            free_SV_coef.append(coef[0][i])
    return free_SV, free_SV_coef


def print_info(clf):
    print clf.coef_
    print clf.n_support_
    print clf.support_vectors_

def linear_kernel(label, data, binary, C):
    new_label = which_binary(label, binary)
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(data, new_label)
    return clf
    #return cal_w_dis(clf)
    #print_info(clf)
    #print clf.coef_

def poly_kernel(label, data, binary, degree, coef, gamma, C):
    new_label = which_binary(label, binary)
    clf = svm.SVC(kernel='poly', degree=degree, coef0=coef, gamma=gamma, C=C)
    clf.fit(data, new_label)
    return clf
    #print_info(clf)

def gaussian_kernel(label, data, binary, gamma, C):
    new_label = which_binary(label, binary)
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
    clf.fit(data, new_label)
    return clf
    #print_info(clf)

