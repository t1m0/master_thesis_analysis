import json
from src.file_handling import extract_subject, extract_simple_file_name
from src.accelerometer import calc_magnitude

def process_spiral_file(file_name):
    file = open(file_name)
    json_data = json.load(file)
    file.close()
    accelerations = json_data['drawing']['accelerations']
    
    if len(accelerations) <= 0: 
        return None

    data = []

    subject = extract_subject(file_name)
    simple_file_name = extract_simple_file_name(file_name)
    hand = json_data['hand']
    device = json_data['device']
    uuid = json_data['drawing']['uuid']
    first_order_smoothness =json_data['result']["firstOrderSmoothness"]
    second_order_smoothness = json_data['result']["secondOrderSmoothness"]
    thightness = json_data['result']["thightness"]
    zero_crossing_rate = json_data['result']["zeroCrossingRate"]

    for acceleration in accelerations:
        x=acceleration['xAxis']
        y=acceleration['yAxis']
        z=acceleration['zAxis']
        time_stamp=acceleration['timeStamp']
        mag=calc_magnitude(x,y,z)
        pandas_row = {
            'subject': subject,
            'file': simple_file_name,
            'uuid':uuid,
            'hand': hand,
            'device': device,
            'first_order_smoothness':first_order_smoothness,
            'second_order_smoothness':second_order_smoothness,
            'thightness':thightness,
            'zero_crossing_rate':zero_crossing_rate,
            'time_stamp':time_stamp,
            'x':x,
            'y':y,
            'z':z,
            'mag':mag
        }
        data.append(pandas_row)
    return data