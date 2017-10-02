import numpy as np
import sys
def MAPE(ans, pred):
    ans = np.array(ans, dtype = np.float)
    pred = np.array(pred, dtype = np.float)
    result = np.sum((np.abs(ans - pred) / ans)) / len(pred)
    return result

def readdata(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
        rawdata = rawdata.split('\n')
        rawdata.pop()
    return rawdata

if __name__ == "__main__":
    prefix = "/home/kirayue/KDD/model/" + sys.argv[2]
    ans = readdata(prefix + '/valid_ans')
    pred = readdata(prefix + '/valid')
    result = MAPE(ans, pred)
    print('valid:{}'.format(result))
