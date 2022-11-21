import os
import re

def ensure_subfolder_exists(file_name):
    # with this select we drop the filename and keep the folder strcuture
    m = re.search('\.\/.*\/', file_name)
    if m:
        found = m.group(0)
        if not os.path.exists(found):
            os.makedirs(found)

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