from datetime import datetime
import cal_avg_volume as statistic
import sys
import numpy as np
global_var = {
        "input": ['/home/kirayue/KDD/data/scripts/training_20min_avg_volume.csv', '/home/kirayue/KDD/data/weather (table 7)_training_update.csv'],
        "model": sys.argv[2],
        'stop': sys.argv[4]
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

def binary_encode(num, position):
    encode_array = np.zeros(num, dtype = np.int)
    encode_array[position] = 1
    return str(tuple(encode_array))

def produceData(data, weather, avg_volume):
    traindata = []
    for key, value in sorted(data.items()):
        starttime = datetime.strptime(value["starttime"], '%Y-%m-%d %H:%M:%S')
        endtime = datetime.strptime(value["endtime"], '%Y-%m-%d %H:%M:%S')
        ### check for holliday
        holliday = '0'
        if datetime(2016, 9, 15) <= starttime <= datetime(2016, 9, 17) or datetime(2016, 10, 1) <= starttime <= datetime(2016, 10, 7):
            holliday = '1'
        if starttime.hour == 8 or starttime.hour == 9 or starttime.hour == 17 or starttime.hour == 18:
            print(key)
            if starttime.hour == 8 or starttime.hour == 9: 
                v_key = key[:15] + '07:40:00'
                c_hour = str(starttime.hour - 7)
                w_key1 = key[4:14] + '-6'
                w_key2 = key[4:14] + '-9'
            elif starttime.hour == 17 or starttime.hour == 18: 
                v_key = key[:15] + '16:40:00'
                c_hour = str(starttime.hour - 14)
                w_key1 = key[4:14] + '-15'
                w_key2 = key[4:14] + '-18'
            volume = getvolume(data, v_key) 
            weekday = str(starttime.weekday())
            tollid = value["tollgate_id"]
            #tollid = binary_encode(3, int(tollgate_id) - 1)
            direction = value["direction"]
            c_minute = str(starttime.minute / 20)
            bv_key = "{}-{}-{}-{}-{}".format(value['tollgate_id'], value['direction'], weekday, c_hour, c_minute)
            before_avg_volume = avg_volume[bv_key] 
            w1 = weather[w_key1]['precipitation'] if w_key1 in weather else '0'
            w2 = weather[w_key2]['precipitation'] if w_key2 in weather else '0'
            if global_var["model"] == 'rfr':
                rowdata = "{} {} {} {} {} {} {} {} {} {} {}".format(value['volume'], weekday, tollid, volume, direction, c_hour, c_minute,  holliday, w1 , w2, before_avg_volume)
            elif global_var["model"] == 'svr':
                rowdata = "{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{} 9:{} 10:{}".format(value['volume'], weekday, tollid, volume, direction, c_hour, c_minute,  holliday, w1 , w2, before_avg_volume)
            else: 
                print("Fuck")
                exit()
            traindata.append(rowdata)
    return traindata
if __name__ == '__main__':
    rawweather = readRawData(global_var['input'][1])
    rawdata = readRawData(global_var['input'][0])
    weather = preprocessWeather(rawweather)
    data = preprocessData(rawdata)
    avg_volume = statistic.avg_volume(data, global_var['stop'])
    traindata = produceData(data, weather, avg_volume)
    prefix = '/home/kirayue/KDD/model/'+ global_var['model']
    with open(prefix + '/train_data', 'w') as f:
        f.write('\n'.join(traindata))
