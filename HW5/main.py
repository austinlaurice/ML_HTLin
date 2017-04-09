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
        E_in_list.append(SVM.error_0_1(label, data, clf))
    utils.curve(C_list_log, E_in_list, '12.png', 'log(C)', 'E_in')

    # question 13
    SV_num_list = []
    #label, data = utils.load_data('features.train')
    for C in C_list_log:
        clf = SVM.poly_kernel(label, data, 8.0, 2, 1, 10**C)
        SV_num_list.append(SVM.SV_num(clf))
    utils.curve(C_list_log, SV_num_list, '13.png', 'log(C)', '#SV')
    
    '''
    # question 14
    C_list_log = [-3, -2, -1, 0, 1,]
    label, data = utils.load_data('features.train')
    for C in C_list_log:
        clf = SVM.gaussian_kernel(label, data, 0.0, 80, 10^C)
        free_SV = SVM.free_SV(clf)
    '''
