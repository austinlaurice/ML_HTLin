from data_processing import process_data, histogram_2
from Logistic_regression import *
from Linear_regression import *
from math import exp

if __name__ == '__main__':
    X_train, Y_train = process_data('./hw3_train.dat')
    X_test, Y_test = process_data('./hw3_test.dat')

    # Q11
    eta = 0.001
    T = 2000
    w = Fixed_rate_Logistic_regression_GD(X_train, Y_train, eta, T)
    correct_rate = error_test_0_1(w, X_test, Y_test)
    error_rate = 1 - correct_rate
    print 'result of Q11'
    print w
    print ("error_rate: %f") % (error_rate)

    # Q12
    w = Fixed_rate_Stochastic_Logistic_regression_GD(X_train, Y_train, eta, T)
    correct_rate = error_test_0_1(w, X_test, Y_test)
    error_rate = 1 - correct_rate
    print 'result of Q12'
    print w
    print ("error_rate: %f") % (error_rate)

    # Q13
    X_train, Y_train = process_data('./hw4_train.dat')
    X_test, Y_test = process_data('./hw4_test.dat')
    Lambda = 1.126
    w = Ridge_Regression(X_train, Y_train, Lambda)
    correct_rate_1 = error_test_0_1(w, X_train, Y_train)
    correct_rate_2 = error_test_0_1(w, X_test, Y_test)
    error_rate_1 = 1 - correct_rate_1
    error_rate_2 = 1 - correct_rate_2
    print 'result of Q13'
    print ("E_in: %f") % (error_rate_1)
    print ("E_out: %f") % (error_rate_2)

    # Q14, 15
    Lambda_list = range(-10, 3)
    E_in = []
    E_out = []
    for L in Lambda_list:
        w = Ridge_Regression(X_train, Y_train, pow(10, L))
        E_in.append((1-error_test_0_1(w, X_train, Y_train)))
        E_out.append((1-error_test_0_1(w, X_test, Y_test)))
    print E_in
    print E_out
    histogram_2(Lambda_list, E_in, 'question_13.png', 'log_10(lambda)', 'E_in')
    histogram_2(Lambda_list, E_out, 'question_14.png', 'log_10(lambda)', 'E_out')

    # Q16


