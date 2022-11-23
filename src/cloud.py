import os
import boto3
from azure.storage.blob import BlobServiceClient
from src.file_handling import delete, ensure_subfolder_exists

bucket_name = 'master-thesis-t1m0'

account_url = "https://t1m0storageaccount.blob.core.windows.net"
container_name = "master-thesis-data"

def download(traget_folder, cloud="azure"):
    delete(traget_folder)
    os.makedirs(traget_folder, exist_ok=True)
    try:
        if cloud == "azure":
            _download_azure(traget_folder)
        elif cloud == "aws":
            _download_aws(traget_folder)
        else:
            print("WTF")


    except Exception as ex:
        print('Exception:')
        print(ex)

def _download_azure(traget_folder):
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    container_client = blob_service_client.get_container_client(container_name)
    # List the blobs in the container
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        src_file_name = blob.name
        if '.json' in src_file_name:
            target_file_name = traget_folder + "/" + src_file_name
            ensure_subfolder_exists(target_file_name)
            with open(file=target_file_name, mode="wb") as download_file:
                download_file.write(container_client.download_blob(blob.name).readall())

def _download_aws(traget_folder):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for file in bucket.objects.filter():
        src_file_name = file.key
        if '.json' in src_file_name:
            target_file_name = traget_folder + target_file_name
            ensure_subfolder_exists(target_file_name)
            bucket.download_file(src_file_name, target_file_name)
