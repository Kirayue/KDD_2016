from datetime import datetime, timedelta
import cal_avg_volume as statistic 
import sys
import numpy as np
global_var = {
        "input": ['/home/kirayue/KDD/data/scripts/training_20min_avg_volume.csv', '/home/kirayue/KDD/data/weather (table 7)_training_update.csv'],
        "valid":{
            'start': datetime.strptime(sys.argv[1], '%Y-%m-%d'),
            'end': datetime.strptime(sys.argv[2], '%Y-%m-%d')
            },
        "after_time": int(sys.argv[3]), # 0, 20, 40, 60, 80, 100
        'part': int(sys.argv[4])
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
    encode_array[int(position) - 1] = 1
    return encode_array

def produceData(data, avg_volume):
    traindata = []
    validdata = []
    trainanswer = []
    validanswer = []
    diff_mins = [120, 100, 80, 60, 40, 20]
    for key, value in sorted(data.items()):
        starttime = datetime.strptime(value["starttime"], '%Y-%m-%d %H:%M:%S')
        endtime = datetime.strptime(value["endtime"], '%Y-%m-%d %H:%M:%S')
        sequence_data = []
        prefix = key[:4]
        ### check for holliday
        holiday = 0
        if datetime(2016, 9, 15) <= starttime <= datetime(2016, 9, 17) or datetime(2016, 10, 1) <= starttime <= datetime(2016, 10, 7):
            holiday = 1
        if ((starttime.hour == 8 or starttime.hour == 17) and starttime.minute == global_var['after_time'] and global_var['part'] == 1) or ((starttime.hour == 9 or starttime.hour == 18) and starttime.minute == global_var['after_time'] and global_var['part'] == 2):
            print(starttime.strftime('%Y-%m-%d %H:%M:%S'))
        #if (starttime.hour == 8 or starttime.hour == 9 or starttime.hour == 17 or starttime.hour == 18) and starttime.minute == global_var['after_time']:
            if starttime.hour == 8 or starttime.hour == 9:
                middletime = datetime.strptime('{}-{}-{} 08:00:00'.format(starttime.year, starttime.month, starttime.day), '%Y-%m-%d %H:%M:%S')
            elif starttime.hour == 17 or starttime.hour == 18:
                middletime = datetime.strptime('{}-{}-{} 17:00:00'.format(starttime.year, starttime.month, starttime.day), '%Y-%m-%d %H:%M:%S')
            for diff_min in diff_mins:
                before_date = middletime - timedelta(minutes=diff_min) ### starttime => middletime
                vkey = prefix + before_date.strftime('%Y-%m-%d %H:%M:%S')
                tollgate_id = binary_encode(3, int(key[0]))
                direction = binary_encode(2, int(key[2]) - 1)
                weekday = binary_encode(7, before_date.isoweekday())
                en_holiday = binary_encode(2, holiday)
                if starttime.hour == 8 or starttime.hour == 9: 
                    c_hour = str(before_date.hour - 5) ####
                elif starttime.hour == 17 or starttime.hour == 18: 
                    c_hour = str(before_date.hour - 12) ####
                hour = binary_encode(4, c_hour) ###########4(8)#############
                minute = binary_encode(3, before_date.minute/20 + 1) 
                b_key = "{}-{}-{}-{}-{}".format(key[0], key[2], before_date.weekday(), c_hour, before_date.minute/20)
                volume = data[vkey]['volume'] if vkey in data else avg_volume[b_key] 
                rowdata = np.concatenate((tollgate_id, direction, weekday, hour, minute, en_holiday), axis = 0)
                rowdata = np.append(rowdata, float(volume))
                sequence_data.append(rowdata)
            if global_var["valid"]["start"] <= starttime <= global_var["valid"]["end"]:
                validdata.append(sequence_data)
                validanswer.append(float(value['volume']))
            else:
                traindata.append(sequence_data)
                trainanswer.append(float(value['volume']))
    return np.array(traindata), np.array(trainanswer), np.array(validdata), np.array(validanswer)

if __name__ == '__main__':
    rawdata = readRawData(global_var['input'][0])
    data = preprocessData(rawdata)
    avg_volume = statistic.avg_volume(data, sys.argv[1])
    traindata, trainanswer, validdata, validanswer = produceData(data, avg_volume)
    prefix = '/home/kirayue/KDD/model/rnn/data/' + str(global_var['after_time']) + '_' + str(global_var['part'])+ '_'
    np.save(prefix + 'train_data', traindata)
    print('Train data shape: ' + str(traindata.shape))
    np.save(prefix + 'train_data_ans', trainanswer)
    print('Train answer data shape: ' + str(trainanswer.shape))
    np.save(prefix + 'valid_data', validdata)
    print('Valid data shape: ' + str(validdata.shape))
    np.save(prefix + 'valid_data_ans', validanswer)
    print('Valid answer data shape: ' + str(validanswer.shape))
