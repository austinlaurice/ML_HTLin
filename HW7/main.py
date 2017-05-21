import utils
import models

if __name__ == '__main__':

    features_train = utils.load_data('hw3_train.dat')
    features_test = utils.load_data('hw3_test.dat')

    # question 7
    alpha_list, g_list, _, _ = models.adaboost(features_train, 300)
    g_err_list = models.predict_with_g(features_train, g_list)
    print "question 7"
    print "Ein(g1) is " + str(g_err_list[0])
    print "alpha1 is " + str(alpha_list[0])
    utils.curve(range(1, 301), g_err_list, '7.png', 't', 'Ein')

    # question 9
    alpha_list, g_list, _, _ = models.adaboost(features_train, 300)
    G_err_list = models.predict_with_G(features_train, g_list, alpha_list)
    print "question 9"
    print "Ein(G) is " + str(G_err_list[299])
    utils.curve(range(1, 301), G_err_list, '9.png', 't', 'Ein')

    # question 10
    _, _, U_list, _ = models.adaboost(features_train, 300)
    print "question 10"
    print "U2 is " + str(U_list[1])
    print "UT is " + str(U_list[299])
    utils.curve(range(1, 301), U_list[:300], '10.png', 't', 'U')
    
    # question 11(don't know why it is different from the answer)
    _, _, _, epsilon_list = models.adaboost(features_train, 300)
    print "question 11"
    print "min epsilon is " + str(min(epsilon_list))
    utils.curve(range(1, 301), epsilon_list, '11.png', 't', 'epsilon')

    # question 12
    alpha_list, g_list, _, _ = models.adaboost(features_train, 300)
    g_err_list = models.predict_with_g(features_test, g_list)
    print "question 12"
    print "Eout(g1) is " + str(g_err_list[0])
    utils.curve(range(1, 301), g_err_list, '12.png', 't', 'Ein')

    # question 13
    alpha_list, g_list, _, _ = models.adaboost(features_train, 300)
    G_err_list = models.predict_with_G(features_test, g_list, alpha_list)
    print "question 13"
    print "Eout(G) is " + str(G_err_list[299])
    utils.curve(range(1, 301), G_err_list, '13.png', 't', 'Eout')
   


    data_train  = utils.load_data_tree('hw3_train.dat')
    data_test  = utils.load_data_tree('hw3_test.dat')
   
    # question 15
    root = models.DecisionTree(data_train)    
    DecisionTree_print(root)



