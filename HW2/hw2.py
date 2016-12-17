from data_processing import process_data, generate_data
from Decision_stump import one_dimension_decision_stump, multi_dimension_decision_stump, check_accuracy


if __name__ == '__main__':
    #Q17, Q18
    E_in_list = []
    E_out_list = []
    for i in range(5000):
        X, Y = generate_data(10, 5)
        score, s, theta = one_dimension_decision_stump(X, Y)
        E_in_list.append(float(score)/10)
        E_out_list.append(Out_of_sample_error(s, theta))
    histogram(E_in_list, 'qustion 17', 'in sample error', 'frequency')
    print "Question 17: average in sample error: %f" % (sum(E_in_list)/5000)
    histogram(E_out_list, 'qustion 18', 'out of sample error', 'frequency')
    print "Question 18: average out of sample error: %f" % (sum(E_out_list)/5000)
    
    #Q19
    X_train, Y_train = process_data('./hw2_train.dat')
    X_test, Y_test = process_data('./hw2_test.dat')
    best_record, s, theta = multi_dimension_decision_stump(X_train, Y_train)
    print "Qustion 19: h = %d * sign(x - %f), in sample error: %f" % (s, theta, float(best_record)/len(Y))
    accuracy = check_accuracy(s, theta, X_test, Y_test)
    print "Qustion 20: out of sample error: %f" % accuracy
