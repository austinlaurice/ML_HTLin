import numpy
import operator
import math

leaf_count = 1


def weighted_0_1(feature, U, t, s):
    count = 0
    for i in xrange(len(feature)):
        if (feature[i][0] >= t and feature[i][1] != s) or (feature[i][0] < t and feature[i][1] == s):
            count += 1 * U[feature[i][2]]
    return count

def decision_stump(U, features):
    '''
    g = (s, i, theta)
    '''
    best_g = [0, 0, 0]
    best_error = 100
    n = 0
    for feature in features:
        error_pos = weighted_0_1(feature, U, feature[0][0], 1)
        error_neg = weighted_0_1(feature, U, feature[0][0], -1)
        threshold_list = []
        threshold_list.append(feature[0][0])
        error_pos_list = []
        error_pos_list.append(error_pos)
        error_neg_list = []
        error_neg_list.append(error_neg)
        for i in range(len(feature) - 1):
            threshold = (feature[i][0] + feature[i+1][0])/2
            if feature[i][1] == 1:
                error_pos += 1 * U[feature[i][2]]
                error_neg -= 1 * U[feature[i][2]]
            elif feature[i][1] == -1:
                error_pos -= 1 * U[feature[i][2]]
                error_neg += 1 * U[feature[i][2]]
            threshold_list.append(threshold)
            error_pos_list.append(error_pos)
            error_neg_list.append(error_neg)
        index_pos, value_pos = min(enumerate(error_pos_list), key=operator.itemgetter(1))
        index_neg, value_neg = min(enumerate(error_neg_list), key=operator.itemgetter(1))
        if value_pos < value_neg and value_pos < best_error:
            best_g = [1, n, threshold_list[index_pos]]
            best_error = value_pos
        if value_pos > value_neg and value_neg < best_error:
            best_g = [-1, n, threshold_list[index_neg]]
            best_error = value_neg
        n += 1
    return best_g

def update_U(g, U, features):
    feature = features[g[1]]
    t = g[2]
    s = g[0]
    sum_U = sum(U)
    count = 0
    for i in xrange(len(feature)):
        if (feature[i][0] >= t and feature[i][1] != s) or (feature[i][0] < t and feature[i][1] == s):
            count += 1 * U[feature[i][2]]
    epsilon = count/float(sum_U)
    diamond = math.sqrt((1-epsilon)/epsilon)
    # update U
    for i in xrange(len(feature)):
        if (feature[i][0] >= t and feature[i][1] != s) or (feature[i][0] < t and feature[i][1] == s):
            U[feature[i][2]] = U[feature[i][2]] * diamond
        else:
            U[feature[i][2]] = U[feature[i][2]] / diamond
    return U, diamond, epsilon


def adaboost(features, T):
    # already sorted
    alpha_list = []
    g_list = []
    U_list = []   # for question 10
    epsilon_list = [] #for question 11
    U = [(1/float(len(features[0])))] * (len(features[0]))
    U_list.append(sum(U))
    for t in xrange(T):
        g = decision_stump(U, features)
        U, diamond_t, epsilon_t = update_U(g, U, features)
        U_list.append(sum(U))
        epsilon_list.append(epsilon_t)
        alpha_list.append(math.log(diamond_t))
        g_list.append(g)
    return (alpha_list, g_list, U_list, epsilon_list)

def predict_with_g(features, g_list):
    e_list = []
    for s, i, t in g_list:
        count = 0
        feature = features[i]
        for j in xrange(len(feature)):
            if (feature[j][0] >= t and feature[j][1] != s) or (feature[j][0] < t and feature[j][1] == s):
                count += 1
        e_list.append(float(count)/len(feature))
    return e_list

def predict_with_G(features, g_list, alpha_list):
    e_list = []
    last = [0] * len(features[0])
    # total T rounds
    for c in xrange(len(alpha_list)):
        s, i, t = g_list[c]
        feature = features[i]
        err = 0
        for j in feature:
            # update last
            if j[0] >= t:
                last[j[2]] += alpha_list[c] * s
            elif j[0] < t:
                last[j[2]] += alpha_list[c] * s * (-1)
            # see error or not
            if numpy.sign(last[j[2]]) != j[1]:
                err += 1
        e_list.append(float(err)/len(feature))
    return e_list





def cal_gini_index(data, class_values):
    s = 0
    gini = 0.0
    if len(data) == 0:
        gini = 0
    else:
        for c in class_values:
            gini += (([d[-1] for d in data].count(c)) / float(len(data)))**2
    return gini

def split_data(data, index, boundary):
    left = list()
    right = list()
    for d in data:
        if d[index] >= boundary:
            right.append(d)
        else:
            left.append(d)
    return left, right

def get_split(data):
    class_values = list(set(d[-1] for d in data))
    best_gini = 1000000
    best_index = 0
    best_boundary = 0
    best_left = None
    best_right = None
    #best_group = None
    for i in xrange(len(data[0]) - 1):
        data = sorted(data, key=lambda k: k[i])
        for j in range(len(data)-1):
            boundary = (data[j][i] + data[j+1][i])/2
            left, right = split_data(data, i, boundary)
            gini = 1 - len(left)*cal_gini_index(left, class_values) - len(right)*cal_gini_index(right, class_values)
            if gini < best_gini:
                best_gini = gini
                best_index = i
                best_boundary = boundary
                best_left = left
                best_right = right
    return {'index': best_index, 'boundary':best_boundary, 'groups': (best_left, best_right)}

def to_terminal(groups):
    outcomes = [row[-1] for row in groups]
    return max(set(outcomes), key=outcomes.count)

def check_termination(data):
    x1 = set()
    x2 = set()
    y = set()
    for d in data:
        x1.add(d[0])
        x2.add(d[1])
        y.add(d[2])
    if (len(x1) == 1 and len(x2) == 1) or len(y) == 1:
        return True
    else:
        return False

def split(node, depth):
    global leaf_count
    left, right = node['groups']
    del node['groups']
    if check_termination(left):
        node['left'] = (to_terminal(left), leaf_count)
        leaf_count += 1
    else:
        node['left'] = get_split(left)
        split(node['left'], depth+1)
    if check_termination(right):
        node['right'] = (to_terminal(right), leaf_count)
        leaf_count += 1
    else:
        node['right'] = get_split(right)
        split(node['right'], depth+1)
    return

def DecisionTree_predict_sub(node, d, prune):
    if d[node['index']] < node['boundary']:
        if isinstance(node['left'], dict):
            return DecisionTree_predict_sub(node['left'], d, prune)
        else:
            if not prune:
                return node['left'][0]
            else:
                if node['left'][1] == prune:
                    if isinstance(node['right'], dict):
                        return DecisionTree_predict_sub(node['right'], d, prune)
                    else:
                        return node['right'][0]
                else:
                    return node['left'][0]
    else:
        if isinstance(node['right'], dict):
            return DecisionTree_predict_sub(node['right'], d, prune)
            '''
            else:
                tmp = DecisionTree_predict_sub(node['right'], d, prune)
                # leaf below is pruned
                if tmp == 100:
                    if isinstance(node['left'], dict):
                        return DecisionTree_predict_sub(node['left'], d, prune)
                    else:
                        return node['left'][0]
                else:
                    return tmp
            '''
        else:
            if not prune:
                return node['right'][0]
            else:
                if node['right'][1] == prune:
                    if isinstance(node['left'], dict):
                        return DecisionTree_predict_sub(node['left'], d, prune)
                    else:
                        return node['left'][0]
                else:
                    return node['right'][0]



def DecisionTree_predict(node, data, prune=None):
    result = []
    for d in data:
        result.append(DecisionTree_predict_sub(node, d, prune))
    return result

def DecisionTree_print(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %f]' % ((depth*2*' ', (node['index']+1), node['boundary'])))
        DecisionTree_print(node['left'], depth+1)
        DecisionTree_print(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*2*' ', node[0])))


def DecisionTree(data_train):
    global leaf_count
    leaf_count = 1
    root = get_split(data_train)
    split(root, 1)
    return root, leaf_count

def error_0_1(p, ans):
    c = 0
    for i in xrange(len(p)):
        if p[i] != ans[i]:
            c += 1
    return c/float(len(p))

def random_sample(data, N):
    
    




def RandomForest(data_train, tree_amount):
    root_list = []
    for i in range(tree_amount):
        data_train_sub = random_sample(data_train, len(data_train))
        root_list.append(DesicionTree(data_train))
    return root_list
