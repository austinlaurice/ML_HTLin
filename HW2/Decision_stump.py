import numpy as np
import random
from data_processing import sort_x_y


# Check only one instance (multi-dimension)
def check_correctness(s, theta, X, y):
    # zero is considered as correct
    correct = 0
    for i in range(len(X)):
        ans = s * np.sign(X[i]-theta)
        if (ans >= 0 and y == 1) or (ans <= 0 and y == -1):
            correct += 1
    if correct == len(X):
        return 1
    return 0

def check_accuracy(s, theta, X, Y):
    correct = 0
    for i in range(len(Y)):
        correct += check_correctness(s, theta, [X[i]], Y[i])
    return float(correct)/(X.shape[0])
    
def one_dimension_decision_stump(X, Y):
    X, Y = sort_x_y(X, Y)
    best_record = 0
    s = 1
    theta = X[0]
    for x in X:
        theta_now = x
        for s_now in [1, -1]:
            rec = 0
            for i in range(len(X)):
                rec += check_correctness(s_now, theta_now, [X[i]], Y[i])
            if rec > best_record:
                s = s_now
                theta = theta_now
                best_record = rec
    return best_record, s, theta

def multi_dimension_decision_stump(X, Y):
    X_trans = np.transpose(X)
    best_record = 0
    possibles = []
    cnt = 0
    for x in X_trans:
        a, s, theta = one_dimension_decision_stump(x, Y)
        if a > best_record:
            best_record = a
            possibles = [(s, theta, cnt)]
        elif a == best_record:
            possibles.append((s, theta, cnt))
        cnt += 1
    w = 0
    if len(possibles) > 1:
        w = random.randint(0, len(possibles)-1)
    return best_record, possibles[w][0], possibles[w][1], possibles[w][2]

def Out_of_sample_error(s, theta):
    return 0.5 + 0.3 * s * (abs(theta) - 1)
    #if s > 0:
    #    return 0.2 + 0.3 * abs(theta)
    #else:
    #    return 0.8 - 0.3 * abs(theta)


