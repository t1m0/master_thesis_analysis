import os
import re

def _drop_file_name(file_path):
    # with this select we drop the filename and keep the folder strcuture
    m = re.search('\.\/.*\/', file_path)
    if m:
        return m.group(0)
    else:
        None

def _reduce_forward_slashes(file_path):
    updated_file_path = file_path.replace('//','/')
    if updated_file_path.endswith('/'):
        updated_file_path = updated_file_path[:-1]
    return updated_file_path

def ensure_subfolder_exists(file_path):
    reduced_path = _drop_file_name(file_path)
    if reduced_path != None and not os.path.exists(reduced_path):
        os.makedirs(reduced_path)

def delete(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                file_path = os.path.join(path, file)
                if os.path.isdir(file_path):
                    delete(file_path)
                else:
                    os.remove(file_path)
            os.rmdir(path)
        else:
            os.remove(path)

def extract_subject(file_path):
    reduced_path = _drop_file_name(file_path)
    reduced_path = _reduce_forward_slashes(reduced_path)
    split = reduced_path.split('/')
    return split[-1]

def extract_simple_file_name(file_path):
    return os.path.split(file_path)[1]

def process_folder(folder_path, file_function):
    results=[]
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isdir(file_path):
            local_results = process_folder(file_path,file_function)
            results = results + local_results
        else:
            local_results = file_function(file_path)
            if local_results != None:
                results = results + local_results
            else:
                print(f'Skipping \'{file_path}\' due to missing accelerations')
    return results
