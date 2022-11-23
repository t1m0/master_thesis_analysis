import os
import rsa
from base64 import b64decode

from src.file_handling import delete,ensure_subfolder_exists, extract_simple_file_name

def decrypt_files(src_folder, target_folder):
    delete(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    for file in os.listdir(src_folder):
        file_path = os.path.join(src_folder, file)
        if os.path.isdir(file_path):
            decrypt_files(file_path, target_folder+'/'+file)
        else:
            _decrypt_file(file_path,target_folder)

def _decrypt_file(file_path,target_folder):
    file = open(file_path)
    lines = file.readlines()
    file_name = extract_simple_file_name(file_path)
    targer_file_name = target_folder+'/'+file_name
    ensure_subfolder_exists(targer_file_name)
    with open(targer_file_name, 'a') as the_file:
        for line in lines:
            the_file.write(_decrypt(line))

def _decrypt(value):
    private_key = open('private.pem', 'rb')
    private_key_data = private_key.read()
    pkcs1 = rsa.PrivateKey.load_pkcs1(private_key_data)
    base64_decoded_value = b64decode(value)
    decrypted = rsa.decrypt(base64_decoded_value, pkcs1)
    return decrypted.decode("utf-8") 
    

