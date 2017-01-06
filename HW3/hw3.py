from data_processing import process_data, histogram_2
from Logistic_regression import *
from Linear_regression import *
from math import exp


if __name__ == '__main__':
    """
    X_train, Y_train = process_data('./hw3_train.dat')
    X_test, Y_test = process_data('./hw3_test.dat')
    # Q11
    eta = 0.001
    T = 2000
    w = Fixed_rate_Logistic_regression_GD(X_train, Y_train, eta, T)
    correct_rate = error_test_0_1(w, X_test, Y_test)
    error_rate = 1 - correct_rate
    print 'result of question 11'
    print w
    print ("error_rate: %f") % (error_rate)

    # Q12
    w = Fixed_rate_Stochastic_Logistic_regression_GD(X_train, Y_train, eta, T)
    correct_rate = error_test_0_1(w, X_test, Y_test)
    error_rate = 1 - correct_rate
    print 'result of question 12'
    print w
    print ("error_rate: %f") % (error_rate)
    """



    # Q13
    X_train, Y_train = process_data('./hw4_train.dat')
    X_test, Y_test = process_data('./hw4_test.dat')
    Lambda = 1.126
    w = Ridge_Regression(X_train, Y_train, Lambda)
    correct_rate_1 = error_test_0_1(w, X_train, Y_train)
    correct_rate_2 = error_test_0_1(w, X_test, Y_test)
    error_rate_1 = 1 - correct_rate_1
    error_rate_2 = 1 - correct_rate_2
    print 'result of question 13'
    print ("E_in: %f") % (error_rate_1)
    print ("E_out: %f") % (error_rate_2)


    # Q14
    Lambda_list = range(-10, 3)
    E_in = []
    w_list = []
    for L in Lambda_list:
        w_list.append(Ridge_Regression(X_train, Y_train, pow(10, L)))
        E_in.append((1-error_test_0_1(w, X_train, Y_train)))
    best = find_best_lambda(E_in, Lambda_list)
    histogram_2(Lambda_list, E_in, 'question_14.png', 'log_10(lambda)', 'E_in')
    error = 1 - error_test_0_1(w_list[best], X_test, Y_test)
    print 'result of question 14'
    print 'best lambda: %f, out of sample error: %f' % (pow(10, Lambda_list[best]), error)

    # Q15
    Lambda_list = range(-10, 3)
    E_in = []
    E_out = []
    for L in Lambda_list:
        w = Ridge_Regression(X_train, Y_train, pow(10, L))
        E_out.append((1-error_test_0_1(w, X_test, Y_test)))
    best = find_best_lambda(E_out, Lambda_list)
    histogram_2(Lambda_list, E_out, 'question_15.png', 'log_10(lambda)', 'E_out')
    error = 1 - error_test_0_1(w_list[best], X_test, Y_test)
    print 'result of question 15'
    print 'best lambda: %f, out of sample error: %f' % (pow(10, Lambda_list[best]), error)

    # Q16
    X_train_s = X_train[:120]
    Y_train_s = Y_train[:120]
    X_val = X_train[120:]
    Y_val = Y_train[120:]
    E_in = []
    E_out = []
    w_list = []
    for L in Lambda_list:
        w_list.append(Ridge_Regression(X_train_s, Y_train_s, pow(10, L)))
        E_in.append((1-error_test_0_1(w, X_train_s, Y_train_s)))
    best = find_best_lambda(E_in, Lambda_list)
    histogram_2(Lambda_list, E_in, 'question_16.png', 'log_10(lambda)', 'E_in')
    error = 1 - error_test_0_1(w_list[best], X_test, Y_test)
    print 'result of question 16'
    print 'best lambda: %f, out of sample error: %f' % (pow(10, Lambda_list[best]), error)

    # Q17
    X_train_s = X_train[:120]
    Y_train_s = Y_train[:120]
    X_val = X_train[120:]
    Y_val = Y_train[120:]
    E_out = []
    E_val = []
    w_list = []
    for L in Lambda_list:
        w_list.append(Ridge_Regression(X_train_s, Y_train_s, pow(10, L)))
        E_val.append((1-error_test_0_1(w, X_val, Y_val)))
    best = find_best_lambda(E_val, Lambda_list)
    histogram_2(Lambda_list, E_val, 'question_17.png', 'log_10(lambda)', 'E_val')
    error = 1 - error_test_0_1(w_list[best], X_test, Y_test)
    print 'result of question 17'
    print 'best lambda: %f, out of sample error: %f' % (pow(10, Lambda_list[best]), error)

    # Q18
    Lambda_best = Lambda_list[best]
    w = Ridge_Regression(X_train, Y_train, pow(10, L))
    E_in = 1 - error_test_0_1(w, X_train, Y_train)
    E_out = 1 - error_test_0_1(w, X_test, Y_test)
    print 'result of question 18'
    print 'best lambda: %f, in sample error: %f, out of sample error: %f' % (pow(10, Lambda_best), E_in, E_out)

    # Q19
    E_cv = []
    X_cv = [ X_train[(i-1)*40:i*40] for i in range(1, 6)]
    Y_cv = [ Y_train[(i-1)*40:i*40] for i in range(1, 6)]
    for L in Lambda_list:
        error_sum = 0
        for i in range(5):
            index = range(5)
            index.remove(i)
            X_cv_train = X_cv[index[1]]
            Y_cv_train = Y_cv[index[1]]
            for j in index[1:]:
                X_cv_train = np.row_stack((X_cv_train, X_cv[j]))
                Y_cv_train = np.concatenate((Y_cv_train, Y_cv[j]))
            w = Ridge_Regression(X_cv_train, Y_cv_train, pow(10, L))
            error_sum += (1 - error_test_0_1(w, X_cv[i], Y_cv[i]))
        E_cv.append(error_sum/5)
    best = find_best_lambda(E_cv, Lambda_list)
    print 'result of question 19'
    print 'Cross validation error: %f, best lambda: %f' % (E_cv[best], pow(10, Lambda_list[best]))

    # Q20
    Lambda_best = Lambda_list[best]
    w = Ridge_Regression(X_train, Y_train, pow(10, L))
    E_in = 1 - error_test_0_1(w, X_train, Y_train)
    E_out = 1 - error_test_0_1(w, X_test, Y_test)
    print 'result of question 20'
    print 'best lambda: %f, in sample error: %f, out of sample error: %f' % (pow(10, Lambda_best), E_in, E_out)


