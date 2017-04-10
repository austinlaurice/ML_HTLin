import utils
import SVM
import numpy

if __name__ == '__main__':
    
    C_list_log = [-5, -3, -1, 1, 3]
    label, data = utils.load_data('features.train')
    test_label, test_data = utils.load_data('features.test')
    
    # question 11
    w_dis = []
    for C in C_list_log:
        clf = SVM.linear_kernel(label, data, 0.0, 10**C)
        w_dis.append(SVM.cal_w_dis(clf))
    print w_dis
    utils.curve(C_list_log, w_dis, '11.png', 'log(C)', '||w||')
    
    # question 12
    E_in_list = []
    #label, data = utils.load_data('features.train')
    print numpy.unique(label)
    for C in C_list_log:
        clf = SVM.poly_kernel(label, data, 8.0, 2, 1, 1, 10**C)
        E_in_list.append(SVM.error_0_1(utils.which_binary(label, 8.0), data, clf))
    utils.curve(C_list_log, E_in_list, '12.png', 'log(C)', 'E_in')
    
    # question 13
    SV_num_list = []
    #label, data = utils.load_data('features.train')
    for C in C_list_log:
        clf = SVM.poly_kernel(label, data, 8.0, 2, 1, 1, 10**C)
        SV_num_list.append(SVM.SV_num(clf))
    print SV_num_list
    utils.curve(C_list_log, SV_num_list, '13.png', 'log(C)', '#SV')
    
    # question 14
    C_list_log = [-3, -2, -1, 0, 1,]
    dis_list = []
    for C in C_list_log:
        clf = SVM.gaussian_kernel(label, data, 0.0, 80, 10**C)
        free_SV, free_SV_coef = SVM.free_SV(clf, 10**C)
        SV = SVM.get_SV(clf)
        SV_coef = SVM.get_dual_coef(clf)
        dis = SVM.cal_dis(SV, SV_coef[0], free_SV[0])
        dis_list.append(dis)
    utils.curve(C_list_log, dis_list, '14.png', 'log(C)', 'dis')
    
    # question 15
    gamma_list = [0, 1, 2, 3, 4]
    C = 0.1
    E_out_list = []
    for gamma in gamma_list:
        clf = SVM.gaussian_kernel(label, data, 0.0, 10**gamma, C)
        E_out_list.append(SVM.error_0_1(utils.which_binary(test_label, 0), test_data, clf))
    utils.curve(gamma_list, E_out_list, '15.png', 'log(gamma)', 'E_out')

    # question 16
    C = 0.1
    gamma_list = [-1, 0, 1, 2, 3]
    gamma_pick = [0, 0, 0, 0, 0]
    for i in xrange(100):
        val_label, val_data, train_label, train_data = utils.split_data(label, data, 1000)
        E_val_list = []
        for gamma in gamma_list:
            clf = SVM.gaussian_kernel(train_label, train_data, 0.0, 10**gamma, C)
            E_val_list.append(SVM.error_0_1(utils.which_binary(val_label, 0), val_data, clf))
        gamma_pick[E_val_list.index(max(E_val_list))] += 1
    utils.histogram(gamma_list, gamma_pick, '16.png', 'log(gamma)', '#selected')
