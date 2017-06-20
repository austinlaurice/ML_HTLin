import numpy
import operator
import math
import random

leaf_count = 1


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
    data_sub = []
    for i in range(N):
        n = random.randint(0, N-1)
        data_sub.append(data[n])
    return data_sub

def RandomForest(data_train, tree_amount, stump=False):
    root_list = []
    for i in range(tree_amount):
        data_train_sub = random_sample(data_train, len(data_train))
        if not stump:
            root_list.append(DecisionTree(data_train_sub)[0])
        else:
            root_list.append(DecisionTreeStump(data_train_sub)[0])

    return root_list

def DecisionTreeStump(data_train):
    global leaf_count
    leaf_count = 1
    root = get_split(data_train)
    left, right = root['groups']
    root['left'] = (to_terminal(left), leaf_count)
    root['right'] = (to_terminal(right), leaf_count)
    return root, leaf_count
