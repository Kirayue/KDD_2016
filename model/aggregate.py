import numpy as np
import sys

def readfile(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
        rawdata = rawdata.split('\n')
    return rawdata

def preprocessX(x1, x2, x3):
    data = []
    for index, value in enumerate(x1):
        data.append('1:{} 2:{} 3:{}'.format(value, x2[index], x3[index]))
    return data

def preprocessY(y):
    Y = []
    for row in y:
        row = row.split(' ')
        Y.append(row[0])
    return Y

if __name__ == "__main__":
    f1 = readfile('./rfr/pred_train')
    f2 = readfile('./svr/pred_train')
    f2.pop()
    f3 = readfile('./rnn/pred_train')
    print(len(f1))
    print(len(f2))
    print(len(f3))
    feature = preprocessX(f1, f2, f3)
    ans_rawdata = readfile('./rfr/train_data')
    y = preprocessY(ans_rawdata)
    rowdata = []
    for i, v in enumerate(feature):
        rowdata.append("{} {}".format(y[i], v))
        
    with open('aggre_train', 'w') as f:
        f.write('\n'.join(rowdata))
