from datetime import datetime
import cal_avg_volume as statistic 
import numpy as np
import sys
global_var = {
        "input": ['/home/kirayue/KDD/data/scripts/training_20min_avg_volume.csv', '/home/kirayue/KDD/data/scripts/test1_20min_avg_volume.csv', '/home/kirayue/KDD/data/dataSets/testing_phase1/weather (table 7)_test1.csv'],
        "model": sys.argv[2],
        "stop": sys.argv[4],
        }
def readRawData(inputfile):
    with open(inputfile, 'r') as f:
        rawdata = f.read()
    rawdata = rawdata.replace('"', '')
    rawdata = rawdata.split('\n')
    rawdata.pop()   # pop '' in the end of list
    rawdata.pop(0)  # pop column name
    return rawdata

def preprocessWeather(data):
    weather = {}
    for row in data:
        row = row.split(',')
        attr = "{}-{}".format(row[0], row[1])
        if attr not in weather:
            weather[attr] = {
                    "pressure": row[2],
                    "sea_pressure": row[3],
                    "wind_direction": row[4],
                    "wind_speed": row[5],
                    "temperature": row[6],
                    "rel_humidity": row[7],
                    "precipitation": row[8]
                    }
    return weather

def preprocessData(rawdata):
    processeddata = {}
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
    return processeddata

def getvolume(data, key):
    if key in data:
        return data[key]["volume"]
    else:
        arr = [float(v["volume"]) for k, v in data.items() if k[:3] == key[:3] and k[15:] == key[15:] ]
        return sum(arr) / len(arr)

def produceData(rawdata, weather, avg_volume):
    data = []
    sub = []
    pred_combine = ['11', '10', '20', '31', '30']
    hour = ['8', '9', '17', '18']
    label_hour = ['1', '2', '3', '4']
    for i in range(7):
        date = datetime.strptime('2016-10-{}'.format(i + 18), '%Y-%m-%d')
        holliday = '0'
        if datetime(2016, 9, 15) <= date <= datetime(2016, 9, 17) or datetime(2016, 10, 1) <= date <= datetime(2016, 10, 7):
            holliday = '1'
        for index, j in enumerate(hour):  # hour
            for k in range(3):  # minute 
                for e in pred_combine:
                    weekday = str(date.weekday())
                    tollid = e[0]
                    direction = e[1]
                    if j == '8' or j == '9': 
                        w_key1 = date.strftime('%Y-%m-%d') + '-6'
                        w_key2 = date.strftime('%Y-%m-%d') + '-9'
                        v_key = "{}-{}-{} 07:40:00".format(e[0], e[1], date.strftime('%Y-%m-%d'))
                    elif j == '17' or j == '18': 
                        w_key1 = date.strftime('%Y-%m-%d') + '-15'
                        w_key2 = date.strftime('%Y-%m-%d') + '-18'
                        v_key = "{}-{}-{} 16:40:00".format(e[0], e[1], date.strftime('%Y-%m-%d'))
                    hour1 = '0' + j if len(j) == 1 else j
                    hour2 = '0' + j if len(j) == 1 else j
                    if k * 20 == 0:
                        minute1 = '00'
                        minute2 = '20'
                    elif k * 20 + 20 == 60:
                        minute1 = str(k * 20)
                        minute2 = '00'
                        hour2 = '0' + str(int(j) + 1) if len(str(int(j) + 1)) == 1 else str(int(j) + 1)
                    else:
                        minute1 = str(k * 20)
                        minute2 = str(k * 20 + 20)
                    rowsub = '{0},"[{1}-{2}-{3} {4}:{5}:00,{1}-{2}-{3} {8}:{7}:00)",{6}'.format(e[0], date.year, date.month, date.day, hour1, minute1, e[1], minute2, hour2)
                    w1 = weather[w_key1]['precipitation'] if w_key1 in weather else '0'
                    w2 = weather[w_key2]['precipitation'] if w_key2 in weather else '0'
                    volume = getvolume(rawdata, v_key)
                    bv_key = "{}-{}-{}-{}-{}.0".format(tollid, direction, weekday, label_hour[index], k)
                    before_avg_volume = avg_volume[bv_key]
                    if global_var["model"] == 'rfr':
                        rowdata = "0 {} {} {} {} {} {} {} {} {} {}".format(weekday, tollid, volume, direction, label_hour[index], k, holliday, w1, w2, before_avg_volume)
                    elif global_var["model"] == 'svr':
                        rowdata = "0 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} 9:{} 10:{}".format(weekday, tollid, volume, direction, label_hour[index], k, holliday, w1, w2, before_avg_volume)
                    else: 
                        print("Fuck")
                        exit()
                    data.append(rowdata)
                    sub.append(rowsub)
    return data, sub
if __name__ == '__main__':
    rawweather = readRawData(global_var['input'][2])
    rawtestdata = readRawData(global_var['input'][1])
    rawtraindata = readRawData(global_var['input'][0])
    weather = preprocessWeather(rawweather)
    data = preprocessData(rawtraindata + rawtestdata)
    avg_volume = statistic.avg_volume(data, global_var['stop'])
    testdata, sub = produceData(data, weather, avg_volume)
    prefix = '/home/kirayue/KDD/model/' + global_var['model']
    with open(prefix + '/test_data', 'w') as f:
        f.write('\n'.join(testdata))
    with open(prefix + '/sub', 'w') as f:
        f.write('\n'.join(sub))
