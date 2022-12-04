import json
from src.file_handling import extract_subject, extract_simple_file_name
from src.accelerometer import calc_magnitude

def process_calibration_file(file_name):
    file = open(file_name)
    json_data = json.load(file)
    file.close()
    accelerations = json_data['accelerations']
    
    if len(accelerations) <= 0: 
        return None

    data = []

    subject = extract_subject(file_name)
    simple_file_name = extract_simple_file_name(file_name)
    hand = json_data['hand']
    uuid = json_data['uuid']
    device = json_data['device']
    start_time = json_data['startTime']

    for acceleration in accelerations:
        x=acceleration['xAxis']
        y=acceleration['yAxis']
        z=acceleration['zAxis']
        time_stamp=acceleration['timeStamp']
        duration = time_stamp - start_time
        mag=calc_magnitude(x,y,z)
        pandas_row = {
            'subject': subject,
            'uuid': uuid,
            'file': simple_file_name,
            'hand': hand,
            'device': device,
            'time_stamp':time_stamp,
            'duration':duration,
            'x':x,
            'y':y,
            'z':z,
            'mag':mag
        }
        data.append(pandas_row)
    return data