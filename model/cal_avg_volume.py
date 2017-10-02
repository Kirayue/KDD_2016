from datetime import datetime
def avg_volume(data, stop):
    avg_volume = {}
    count = {}
    volume = {}
    candi_hour1 = [6, 7, 8, 9]
    candi_hour2 = [15, 16, 17, 18]
    stopdate = datetime.strptime(stop, '%Y-%m-%d')
    for k, v in data.items():
        date = datetime.strptime(v["starttime"], '%Y-%m-%d %H:%M:%S')
        if date < stopdate:
            weekday = date.weekday()
            if date.hour in candi_hour1:
                c_hour = str(date.hour - 5)
            elif date.hour in candi_hour2:   
                c_hour = str(date.hour - 12)
            else:
                continue
            key = "{}-{}-{}-{}-{}".format(v['tollgate_id'], v['direction'], str(date.weekday()), c_hour, str(date.minute / 20))
            if key in count and key in volume:
                count[key] += 1
                volume[key] += float(v["volume"])
            else:
                count[key] = 1
                volume[key] = float(v['volume'])
    for k, v in volume.items():
        avg_volume[k] = v / count[k]
    return avg_volume
