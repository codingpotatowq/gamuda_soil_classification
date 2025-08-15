import os
import PIL
from PIL import Image


# == directory ==
dir_OA = r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Dataset\OA'
dir_OBOA =  r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Dataset\OBOA'

directory = [
    dir_OA, dir_OBOA
]


# == check image format and convert to jpg ==
def convert_to_jpg(directory, quality=95):
    supported_format = ['.png', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif']

    converted = 0
    errors = 0

    print(f"Processing: {os.path.basename(directory)}")

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_name, file_ext = os.path.splitext(file)

        if file_ext.lower() in supported_format:
            try:
                with Image.open(file_path) as img:
                    print('Size of image before conversion: ', end = "")
                    print(os.path.getsize(file_path))

                    if img.mode != 'RGB':
                        if img.mode in ('RGBA', 'LA', 'P'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[-1])
                            img = background
                        else:
                            img = img.convert('RGB')

                    new_path = os.path.join(directory, f"{file_name}.jpg")
                    img.save(new_path, 'JPEG', quality = quality)

                    os.remove(file_path)
                    converted += 1
                    print(f'{file} converted')

            except Exception as e:
                errors += 1
                print(f'error with {file} : {e}')

    print(f'summary: converted {converted}, errors {errors}')


# == command ==
for folder in directory:
    if os.path.exists(folder):
        convert_to_jpg(folder)
    else:
        print(f'Folder not found: {folder}')
