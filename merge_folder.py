import os
import shutil

source_folder = r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Dataset\OBOA'
destination_folder =r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Dataset\OBOA'

for main_path, folder_name, file_names in os.walk(source_folder):
    '''print(f'main directory: {main_path}')
    print(f'sub-directories: {folder_name}')
    print(f'files: {file_names}')'''

    '''for folder in folder_name:
        folder_path = os.path.join(main_path, folder)
        print(folder_path)'''

    for file in file_names:
        src_file_path = os.path.join(main_path, file)
        print(f'file path: {src_file_path}')
        dest_file_path = os.path.join(source_folder, file)
        print(f'destnation file path: {dest_file_path}')
        shutil.move(src_file_path, dest_file_path)
