import os
import google
from google.cloud import storage
import cv2
import imagehash
import PIL.Image
from datetime import date, timedelta

# == GCS Authentication Setup ==
credentials, project_id = google.auth.default()
client = storage.Client(credentials=credentials, project=project_id)
bucket_name = 'get_sgtunnel_atbm_excavation_image'
bucket = client.get_bucket(bucket_name)

folder = r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Dataset\ALL' # (CHANGE DATASET PATH HERE)
delimiter = '/'

blobs = bucket.list_blobs(delimiter = delimiter)

# == Download ALL Files to Local ==
def download_to_local():
    if not os.path.exists(folder):
        os.makedirs(folder)

    for blob in blobs: 
        blob_name = blob.name
        creation_date = blob.time_created.date()
        ring_split = blob_name.split('_') #get ring number for folder name
        ring_name = ring_split[1]
        tunnel_name = ring_split[0]
        
        tunnel_folder = os.path.join(folder, tunnel_name)
        if not os.path.exists(tunnel_folder): #ring folder organised based on tunnel
            os.makedirs(tunnel_folder)

        ring_folder = os.path.join(tunnel_folder, ring_name) #saved based on ring folder
        if not os.path.exists(ring_folder):
            os.makedirs(ring_folder)

        file_name = os.path.basename(blob_name)
        local_file_path = os.path.join(ring_folder, file_name)

        blob.download_to_filename(local_file_path)
        print(f'Downloaded {file_name}, creation date: {creation_date} to {ring_folder}.')

download_to_local()