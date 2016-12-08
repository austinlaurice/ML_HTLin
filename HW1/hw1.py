from data_processing import process_data, histogram 
from PLA import naive_cyclic_PLA
from collections import Counter

def data_reorder(update):
    update.sort()
    freq = dict(Counter(update))
    update = list(set(update)).sort()
    index = range(update[0], update[-1]+1)
    freq_list = []
    for i in index:
        if i in freq:
            freq_list.append(freq[x])
        else:
            freq_list.append(0)
    return index, freq_list


if __name__ == '__main__':
    X, Y = process_data('./hw1_15_train.dat.txt')
    _, index_record, _ = naive_cyclic_PLA(X, Y)
    print 'updates: %d, index that results in max updates: %d' % (sum(index_record), index_record.argsort()[::-1][0]) 
    
    update = []
    for i in range(2000):
        _, index_record, _ = naive_cyclic_PLA(X, Y, random=True)
        total_update = sum(index_record)
        update.append(total_update)
    update, freq = data_reorder(update)
    histogram(update, freq, 'question16.png')
    
    update = []
    for i in range(2000):
        _, index_record, _ = naive_cyclic_PLA(X, Y, random_ord=True, nu=0.25)
        total_update = sum(index_record)
        update.append(total_update)
    update, freq = data_reorder(update)
    histogram(update, freq, 'question17.png')
