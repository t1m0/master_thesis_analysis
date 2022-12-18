import json
from src.file_handling import extract_subject, extract_simple_file_name
from src.accelerometer import calc_magnitude
from src.file_processing import extract_age_group


def _process_accelerations(base_record, start_time, accelerations):
    data = []
    for acceleration in accelerations:
        record = base_record.copy()
        x = acceleration['xAxis']
        y = acceleration['yAxis']
        z = acceleration['zAxis']
        time_stamp = acceleration['timeStamp']
        duration = time_stamp - start_time
        mag = calc_magnitude(x, y, z)
        record['time_stamp'] = time_stamp
        record['duration'] = duration
        record['x'] = x
        record['y'] = y
        record['z'] = z
        record['mag'] = mag
        data.append(record)
    return data


def process_drift_file(file_name):
    file = open(file_name)
    json_data = json.load(file)
    file.close()
    accelerations = json_data['accelerations']

    if len(accelerations) <= 0:
        return None

    data = []

    subject = extract_subject(file_name)
    age_group = extract_age_group(subject)
    simple_file_name = extract_simple_file_name(file_name)
    uuid = json_data['uuid']
    dominant_hand_device = json_data['dominantDevice']
    non_dominant_hand_device = json_data['nonDominantDevice']

    for hand in accelerations:
        if "dominant" == hand:
            device = dominant_hand_device
        else:
            device = non_dominant_hand_device
        base_record = {
            'subject': subject,
            'age_group': age_group,
            'file': simple_file_name,
            'uuid': uuid,
            'hand': hand,
            'device': device
        }
        hand_accelerations = accelerations[hand]
        if len(hand_accelerations) <= 0:
            return None
        start_time = hand_accelerations[0]['timeStamp']
        data = data + _process_accelerations(base_record,start_time, hand_accelerations)
    return data
