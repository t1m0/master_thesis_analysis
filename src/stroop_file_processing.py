import json
from src.file_handling import extract_subject, extract_simple_file_name
from src.accelerometer import calc_magnitude
from src.file_processing import extract_age_group

def _find_click_number(clicks_time_stamps, current_time_stamp):
    click_number = 0
    for i in range(len(clicks_time_stamps)):
        if clicks_time_stamps[i] > current_time_stamp:
            break
        else:
            click_number = (i+1)
    return click_number

def process_stroop_file(file_path):
    file = open(file_path)
    json_data = json.load(file)
    file.close()
    accelerations = json_data['gameSession']['accelerations']

    if len(accelerations) <= 0: 
        return None

    data = []

    subject = extract_subject(file_path)
    age_group = extract_age_group(subject)
    simple_file_name = extract_simple_file_name(file_path)
    hand = json_data['hand']
    device = json_data['device']
    uuid = json_data['gameSession']['uuid']

    clicks = json_data['gameSession']['clicks']
    start_time = json_data['gameSession']['startTime']
    click_distance_mean = json_data['meanDistance']
    click_distance_std = json_data['distanceStandardDeviation']
    click_success_rate = json_data['successRate']
    clicks_time_stamps = []
    for click in clicks:
        if click['valid']:
            clicks_time_stamps.append(click['timeStamp'])

    for acceleration in accelerations:
        time_stamp = acceleration['timeStamp']
        click_number = _find_click_number(clicks_time_stamps, time_stamp)
        duration = time_stamp - start_time
        x=acceleration['xAxis']
        y=acceleration['yAxis']
        z=acceleration['zAxis']
        mag=calc_magnitude(x,y,z)
        pandas_row = {
            'subject': subject,
            'age_group': age_group,
            'file': simple_file_name,
            'uuid':uuid,
            'hand': hand,
            'device': device,
            'click_distance_mean': click_distance_mean,
            'click_distance_std': click_distance_std,
            'click_success_rate': click_success_rate,
            'click_number':click_number,
            'duration':duration,
            'time_stamp':time_stamp,
            'x':x,
            'y':y,
            'z':z,
            'mag':mag
        }
        data.append(pandas_row)
    return data
