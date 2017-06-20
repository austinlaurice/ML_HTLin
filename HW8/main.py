import utils
import models
import collections

if __name__ == '__main__':
    data_train  = utils.load_data_tree('hw3_train.dat')
    data_test  = utils.load_data_tree('hw3_test.dat')
    root_list = models.RandomForest(data_train, 30000)
    p_in_list = [models.DecisionTree_predict(r, data_train) for r in root_list]
    p_out_list = [models.DecisionTree_predict(r, data_test) for r in root_list]
    ans_in = [d[2] for d in data_train]
    ans_out = [d[2] for d in data_test]
   
    # question 12
    Ein_list = [ models.error_0_1(p, ans_in) for p in p_in_list]
    #ans_dict = collections.Counter(Ein_list)
    #x = sorted([i for i in ans_dict.keys()])
    #y = [ans_dict[i] for i in x]
    #import ipdb; ipdb.set_trace()
    print "question 12"
    utils.histogram(Ein_list, '12.png', 'Ein', 'Count')

    # question 13
    sum_p = [0] * len(p_in_list[0])
    result = []
    for p in p_in_list:
        sum_p = [(x + y)  for (x, y) in zip(p, sum_p)]
        tmp = []
        for a in sum_p:
            if a > 0:
                tmp.append(1)
            else:
                tmp.append(-1)
        result.append(tmp)
    Ein_list = [ models.error_0_1(p, ans_in) for p in result]
    x = range(1, len(Ein_list)+1)
    print "question 13"
    utils.curve(x, Ein_list, '13.png', 't', 'Ein')
    
    # question 14
    sum_p = [0] * len(p_out_list[0])
    result = []
    for p in p_out_list:
        sum_p = [(x + y)  for (x, y) in zip(p, sum_p)]
        tmp = []
        for a in sum_p:
            if a > 0:
                tmp.append(1)
            else:
                tmp.append(-1)
        result.append(tmp)
    Eout_list = [ models.error_0_1(p, ans_out) for p in result]
    x = range(1, len(Eout_list)+1)
    print "question 14"
    utils.curve(x, Eout_list, '14.png', 't', 'Eout')
    
    root_list = models.RandomForest(data_train, 30000, stump=True)
    p_in_list = [models.DecisionTree_predict(r, data_train) for r in root_list]
    p_out_list = [models.DecisionTree_predict(r, data_test) for r in root_list]
    ans_in = [d[2] for d in data_train]
    ans_out = [d[2] for d in data_test]
    
    # question 15
    sum_p = [0] * len(p_in_list[0])
    result = []
    for p in p_in_list:
        sum_p = [(x + y)  for (x, y) in zip(p, sum_p)]
        tmp = []
        for a in sum_p:
            if a > 0:
                tmp.append(1)
            else:
                tmp.append(-1)
        result.append(tmp)
    Ein_list = [ models.error_0_1(p, ans_in) for p in result]
    x = range(1, len(Ein_list)+1)
    print "question 15"
    utils.curve(x, Ein_list, '15.png', 't', 'Ein')

    # question 16
    sum_p = [0] * len(p_out_list[0])
    result = []
    for p in p_out_list:
        sum_p = [(x + y)  for (x, y) in zip(p, sum_p)]
        tmp = []
        for a in sum_p:
            if a > 0:
                tmp.append(1)
            else:
                tmp.append(-1)
        result.append(tmp)
    Eout_list = [ models.error_0_1(p, ans_out) for p in result]
    x = range(1, len(Eout_list)+1)
    print "question 16"
    utils.curve(x, Eout_list, '16.png', 't', 'Eout')

