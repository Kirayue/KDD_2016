from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sys

def readfile(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
        rawdata = rawdata.split('\n')
    return rawdata

def preprocessData(rawdata):
    X = []
    Y = []
    for row in rawdata:
        row = row.split(' ')
        X.append(row[1:])
        Y.append(row[0])
    return np.array(X, dtype = np.float), np.array(Y, dtype = np.float) 

if __name__ == "__main__":
    rawtraindata = readfile('train_data')
    trainX, trainY = preprocessData(rawtraindata)
    rfr = RandomForestRegressor()
    rfr.fit(trainX, trainY)
    rawtestdata = readfile('train_data')
    testX, testY = preprocessData(rawtestdata)
    with open('pred_train', 'w') as f:
        f.write('\n'.join('%0.3f' %pred for pred in rfr.predict(testX)))
