from datetime import datetime
import sys

def readRawData(inputfile):
    with open(inputfile, 'r') as f:
        rawdata = f.read()
    rawdata = rawdata.replace('"', '')
    rawdata = rawdata.split('\n')
    return rawdata

def preprocessData(rawdata, data):
    processeddata = data
    for row in rawdata:
        row = row.split(',')
        attr = "{}-{}-{}".format(row[0], row[3], row[1][1:])
        if attr not in processeddata:
            processeddata[attr] = {
                    "starttime": row[1][1:],
                    "endtime": row[2][:-1],
                    "volume": row[4],
                    "tollgate_id": row[0],
                    "direction": row[3]
                    }
        else: 
            print(attr)

def produceData(data):
    trainpred = []
    for key, value in sorted(data.items()):
        print(key)
        trainpred.append(value['volume'])
    return trainpred 
if __name__ == '__main__':
    data = {}
    #prefix = '/home/kirayue/KDD/model/rnn/temp/{}_{}_{}_{}_trainpred'
    prefix = '/home/kirayue/KDD/model/rnn/temp/{}_{}_{}_trainpred'
    part = [1, 2]
    time = [0, 20, 40]
    iteration = [10000, 15000, 20000, 25000, 30000, 35000]
    #iteration = [30000, 35000, 40000, 45000, 65000, 70000]
    count = 0
    for p in part:
        for t in time:
            #rawdata = readRawData(prefix.format(sys.argv[1], iteration[count],t, p))
            rawdata = readRawData(prefix.format(iteration[count],t, p))
            print(len(rawdata))
            preprocessData(rawdata, data)
            count += 1
    trainpred = produceData(data)
    with open('./aggredata/{}_pred_train'.format(sys.argv[1]), 'w') as f:
        f.write('\n'.join(trainpred))
