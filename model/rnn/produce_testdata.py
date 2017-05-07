from datetime import datetime, timedelta
import cal_avg_volume as statistic 
import numpy as np
import sys
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
        arr = [float(v["volume"]) for k, v in data.items() if k[:3] == key[:3] and k[15:] == key[15:]]
        return sum(arr) / len(arr)

def binary_encode(num, position):
    encode_array = np.zeros(num, dtype = np.int)
    encode_array[int(position) - 1] = 1
    return encode_array

def produceData(rawdata, avg_volume, preddate):
    testdata = []
    sequence_data = []
    diff_mins = [120, 100, 80, 60, 40, 20]
    temp = preddate[4:]
    if preddate[15:17] == '08' or preddate[15:17] == '09':
        temp = temp[:11] + '08:00' + temp[16:]
    else:
        temp = temp[:11] + '17:00' + temp[16:]
    #print(preddate)
    predtime = datetime.strptime(temp, '%Y-%m-%d %H:%M:%S')
    holiday = 0
    prefix = preddate[:4]
    if datetime(2016, 9, 15) <= predtime <= datetime(2016, 9, 17) or datetime(2016, 10, 1) <= predtime <= datetime(2016, 10, 7):
        holiday = 1
    for diff_min in diff_mins:
        before_date = predtime - timedelta(minutes = diff_min)
        vkey = prefix + before_date.strftime('%Y-%m-%d %H:%M:%S')
        tollgate_id = binary_encode(3, int(preddate[0]))
        direction = binary_encode(2, int(preddate[2])- 1)
        weekday = binary_encode(7, before_date.isoweekday())
        en_holiday = binary_encode(2, holiday)
        if predtime.hour == 8 or predtime.hour == 9: 
            c_hour = str(before_date.hour - 5)
        elif predtime.hour == 17 or predtime.hour == 18: 
            c_hour = str(before_date.hour - 12)
        hour = binary_encode(4, c_hour) 
        minute = binary_encode(3, before_date.minute/20 + 1) 
        b_key = "{}-{}-{}-{}-{}".format(preddate[0], preddate[2], before_date.weekday(), c_hour, before_date.minute/20)
        volume = rawdata[vkey]['volume'] if vkey in rawdata else avg_volume[b_key]
        rowdata = np.concatenate((tollgate_id, direction, weekday, hour, minute, en_holiday), axis = 0)
        rowdata = np.append(rowdata, float(volume))
        #rowdata = np.array([float(volume)])
        sequence_data.append(rowdata)
    testdata.append(sequence_data)
    return np.array(testdata)

def produceSub(preddate):
    start_preddate = datetime.strptime(preddate[4:], '%Y-%m-%d %H:%M:%S')
    end_preddate = start_preddate + timedelta(minutes = 20)
    start_time = start_preddate.strftime('%Y-%m-%d %H:%M:%S')
    end_time = end_preddate.strftime('%Y-%m-%d %H:%M:%S')
    rowsub = '{0},"[{1},{2})",{3}'.format(preddate[0], start_time, end_time, preddate[2])
    return rowsub
