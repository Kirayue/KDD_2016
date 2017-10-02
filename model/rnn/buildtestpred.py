from datetime import datetime
import produce_testdata as pt 
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
    testpred = []
    sub = ['tollgate_id,time_window,direction,volume']
    for key, value in sorted(data.items()):
        #print(key)
        preddate = "{}-{}-{}".format(value['tollgate_id'], value['direction'], value['starttime'])
        rowsub = pt.produceSub(preddate)
        rowsub += "," + str(value['volume'])
        sub.append(rowsub)
        testpred.append(value['volume'])
    return testpred, sub
if __name__ == '__main__':
    data = {}
    #prefix = '/home/kirayue/KDD/model/rnn/pred/{}_{}_{}_{}_sub_avg.csv'
    prefix = '/home/kirayue/KDD/model/rnn/pred/phase2_{}_{}_{}_sub_avg.csv'
    part = [1, 2]
    time = [0, 20, 40]
    iteration = [10000, 15000, 10000, 15000, 10000, 18000]
    #iteration = [30000, 35000, 40000, 45000, 65000, 70000]
    count = 0
    for p in part:
        for t in time:
            #rawdata = readRawData(prefix.format(sys.argv[1], iteration[count],t, p))
            rawdata = readRawData(prefix.format(iteration[count],t, p))
            print(len(rawdata))
            preprocessData(rawdata, data)
            count += 1
    testpred, sub = produceData(data)
    with open('./pred/{}_pred_rnn'.format(sys.argv[1]), 'w') as f:
        f.write('\n'.join(testpred))
    with open('./sub/{}_sub.csv'.format(sys.argv[1]), 'w') as f:
        f.write('\n'.join(sub))
