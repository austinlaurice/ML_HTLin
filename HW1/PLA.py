import numpy as np
import random

def check_correctness(weight, X, Y):
    tmp = np.dot(weight, X)
    if tmp == 0:
        tmp = -1
    if np.sign(tmp) != Y[i]:
        return False
    return True

def better_one(w1, w2, X, Y):
    m1 = 0
    m2 = 0
    for i in range(len(Y)):
        if not check_correctness(w1, X[i], Y[i]):
            m1 += 1
        if not check_correctness(w2, X[i], Y[i]):
            m2 += 1
    if m1 > m2:
        return w2
    else:
        return w1

def naive_cyclic_PLA(X, Y, random_ord=False, nu=1):
    weight = np.zeros(len(X[0]), dtype=np.float64)
    index_record = np.zeros(len(Y))
    mistake = 0
    round_cnt = 0
    order = range(len(Y))
    if random_ord:
        random.shuffle(order)
    while True:
        for i in order:
            if check_correctness(weight, X[i], Y[i]):
                weight = weight + nu * Y[i]*X[i]
                mistake += 1
                index_record[i] += 1
            round_cnt += 1
        if mistake == 0:
            break
        mistake = 0
    return weight, index_record, round_cnt

def pocket_PLA(X, Y, updates=50):
    #weight = np.random.rand(len(X[0]))
    weight = np.zeros(len(X[0]), dtype=np.float64)
    update = 0
    while update < updates:
        i = random.randrange(0, len(Y)-1)
        if not check_correctness(weight, X[i], Y[i]):
            tmp = weight + Y[i]*X[i]
            weight = better_one(weight, tmp, X, Y)
            update += 1
    return weight

def test_accuracy(w, X, Y):
    m = 0
    for i in range(len(Y)):
        if not check_correctness(w, X[i], Y[i]):
            m += 1
    return m
