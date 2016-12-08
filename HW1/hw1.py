from data_processing import process_data, histogram
from PLA import naive_cyclic_PLA, pocket_PLA, test_accuracy
from collections import Counter

def data_reorder(update):
    update.sort()
    update = [int(x) for x in update]
    freq = dict(Counter(update))
    update = sorted(list(set(update)))
    index = range(int(update[0]), int(update[-1])+1)
    freq_list = []
    for i in index:
        if i in freq:
            freq_list.append(freq[x])
        else:
            freq_list.append(0)
    print index, freq_list
    return index, freq_list


if __name__ == '__main__':
    X, Y = process_data('./hw1_15_train.dat.txt')

    # Q15
    _, index_record, _ = naive_cyclic_PLA(X, Y)
    print 'question 15: updates: %d, index that results in max updates: %d' % (sum(index_record), index_record.argsort()[::-1][0]) 

    #Q16
    print 'question 16'
    update = []
    for i in range(2000):
        _, index_record, _ = naive_cyclic_PLA(X, Y, random_ord=True)
        total_update = sum(index_record)
        update.append(int(total_update))
    #update, freq = data_reorder(update)
    #histogram(update, freq, 'question16.png')
    histogram(update, 'question16.png', 'number of updates', 'frequency')

    #Q17
    print 'question 17'
    update = []
    for i in range(2000):
        _, index_record, _ = naive_cyclic_PLA(X, Y, random_ord=True, nu=0.25)
        total_update = sum(index_record)
        update.append(int(total_update))
    #update, freq = data_reorder(update)
    #histogram(update, freq, 'question17.png')
    histogram(update, 'question17.png', 'number of updates', 'frequency')

    X_train, Y_train = process_data('./hw1_18_train.dat.txt')
    X_test, Y_test = process_data('./hw1_18_test.dat.txt')

    #Q18
    print 'question 18'
    errors = []
    for i in range(2000):
        weight = pocket_PLA(X_train, Y_train)
        error = test_accuracy(weight, X_test, Y_test)
        errors.append(error)
    histogram(errors, 'question18.png', 'error rate', 'frequency')

    #Q19
    print 'question 19'
    errors = []
    for i in range(2000):
        weight = pocket_PLA(X_train, Y_train, updates=100)
        error = test_accuracy(weight, X_test, Y_test)
        errors.append(error)
    histogram(errors, 'question19.png', 'error rate', 'frequency')

    #Q20
    print 'question 20'
    errors = []
    for i in range(2000):
        weight = pocket_PLA(X_train, Y_train, updates=100, pocket=False)
        error = test_accuracy(weight, X_test, Y_test)
        errors.append(error)
    histogram(errors, 'question20.png', 'error rate', 'frequency')
