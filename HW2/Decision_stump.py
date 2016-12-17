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
        correct += check_correctness(x, Y[i])
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
                rec += check_correctness(s_now, theta_now, X[i], Y[i])
            if rec > best_record:
                s = s_now
                theta = theta_now
                best_record = rec
    return best_record, s, theta            

def multi_dimension_decision_stump(X, Y):
    X_trans = np.transpose(X)
    possibles = []
    for x in X_trans:
        _, s, theta = one_dimension_decision_stump(x, Y)
        possibles.append((s, theta))
    best_record = 0
    s = []
    theta = []
    for p in possibles:
        rec = 0
        for i in range(X.shape[0]):
            rec += check_correctness(p[0], p[1], X[i], Y[i])
        if rec > best_record:
            best_record = rec
            s = [p[0]]
            theta = [p[1]]
        elif rec == best_record:
            s.append(p[0])
            theta.append(p[1])
    if len(s) > 1:
        r = random.randint(0, len(s)-1)
        s = s[r]
        theta = theta[r]
    else:
        s = s[0]
        theta = theta[0]
    return best_record, s, theta

def Out_of_sample_error(s, theta):
    if s > 0:
        return 0.2 + 0.3 * abs(theta)
    else:
        return 0.8 - 0.3 * abs(theta)


