import json
from src.file_handling import extract_subject, extract_simple_file_name
from src.accelerometer import calc_magnitude
from src.file_processing import extract_age_group


def process_slow_move_accelerations_file(file_path):
    file = open(file_path)
    json_data = json.load(file)
    file.close()
    accelerations = json_data['accelerations']

    if len(accelerations) <= 0:
        return None

    data = []

    subject = extract_subject(file_path)
    age_group = extract_age_group(subject)
    simple_file_name = extract_simple_file_name(file_path)
    hand = json_data['hand']
    device = json_data['device']
    uuid = json_data['uuid']
    start_time = json_data['startTime']

    for acceleration in accelerations:
        time_stamp = acceleration['timeStamp']
        duration = time_stamp - start_time
        x = acceleration['xAxis']
        y = acceleration['yAxis']
        z = acceleration['zAxis']
        mag = calc_magnitude(x, y, z)
        pandas_row = {
            'subject': subject,
            'age_group': age_group,
            'file': simple_file_name,
            'uuid': uuid,
            'hand': hand,
            'device': device,
            'duration': duration,
            'time_stamp': time_stamp,
            'x': x,
            'y': y,
            'z': z,
            'mag': mag
        }
        data.append(pandas_row)
    return data


def process_slow_move_touches_file(file_path):
    file = open(file_path)
    json_data = json.load(file)
    file.close()
    movements = json_data['movements']

    if len(movements) <= 0:
        return None

    data = []

    subject = extract_subject(file_path)
    age_group = extract_age_group(subject)
    simple_file_name = extract_simple_file_name(file_path)
    hand = json_data['hand']
    uuid = json_data['uuid']
    start_time = json_data['startTime']
    start_x = json_data['start']['x']
    start_y = json_data['start']['y']
    end_x = json_data['end']['x']
    end_y = json_data['end']['y']

    for movement in movements:
        time_stamp = movement['timeStamp']
        duration = time_stamp - start_time
        x = movement['x']
        y = movement['y']
        pandas_row = {
            'subject': subject,
            'age_group': age_group,
            'file': simple_file_name,
            'uuid': uuid,
            'hand': hand,
            'duration': duration,
            'time_stamp': time_stamp,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'x': x,
            'y': y
        }
        data.append(pandas_row)
    return data
