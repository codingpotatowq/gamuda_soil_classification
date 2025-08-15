'''
1. download only wb files (may 24- june 24) from gcs
2. remove duplicate
3. crop
4. colour correction(??)
5. upload to gcdrive
'''
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

folder = r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Dataset\ALL' # (CHANGE OUTPUT PATH HERE)
supported_name = 'WB'
delimiter = '/'


# == To get 30 days from Today ==
today_date = date.today()
print(f"today: {today_date}")
last_month = today_date - timedelta(days = 30)
print(f"last month: {last_month}")

blobs = bucket.list_blobs(delimiter = delimiter)


# == Download Files with name 'WB' from the Last 30 Days to Local ==
def download_to_local():
    if not os.path.exists(folder):
        os.makedirs(folder)

    for blob in blobs: 
        blob_name = blob.name
        creation_date = blob.time_created.date()
        ring_split = blob_name.split('_') #get ring number for folder name
        ring_name = ring_split[1]
        if supported_name in blob_name:
            print(f'Blob: {blob_name}, creation date: {creation_date}.')
            if last_month <= creation_date <= today_date:
                ring_folder = os.path.join(folder, ring_name) #saved based on ring folder
                if not os.path.exists(ring_folder):
                    os.makedirs(ring_folder)
                file_name = os.path.basename(blob_name)
                local_file_path = os.path.join(ring_folder, file_name)

                blob.download_to_filename(local_file_path)
                print(f'Downloaded {file_name}, creation date: {creation_date} to {ring_folder}.')


# == Find Similar Images and Delete ==
def find_delete_similar_image(folder, similarity = 20):
    image_hashes = {}

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        image = PIL.Image.open(file_path)
        hashed_image = imagehash.phash(image) #convert to hash number

        image_hashes[file_path] = hashed_image
        print(f'Processed {file_name} for duplicates.')
    
    processed = set()
    duplicates = []

    for file_path1, hash1 in image_hashes.items():
        if file_path1 in processed:
            continue

        similar_group = [file_path1]

        for file_path2, hash2 in image_hashes.items():
            if file_path1 != file_path2 and file_path2 not in processed:
                distance = hash1 - hash2 #get the hash difference between 2 images

                print(f'distance: {str(distance)}')

                if distance <= similarity: #if hash difference is below a certain threshold, the 2nd file will be appended
                    similar_group.append(file_path2)

        if len(similar_group) > 1: #groups the similar pictures together and counts the number of groups
            duplicates.append(similar_group)
            processed.update(similar_group)

    print(f'Found {len(duplicates)} groups of similar images.')

    deleted_count = 0

    for i, group in enumerate(duplicates): #enumerate over the groups and only keeps the largest file
        print(f"\nGroup {i+1}: {len(group)} similar images")

        group.sort(key = lambda x: os.path.getsize(x), reverse = True)
        keep_file = group[0]
        delete_files = group[1:]
        print(f'Keeping {os.path.basename(keep_file)}')
        for file in delete_files:
            os.remove(file)
            deleted_count += 1
            print(f'Deleted {os.path.basename(file)}')
            
    return duplicates


# == Preprocess images by cropping ==
def image_preprosessing(file_path, file_name):
    image = cv2.imread(file_path)
    #blurred_image = cv2.GaussianBlur(image, (5, 5), 0) 
    x_start, y_start, x_end, y_end = 346, 181, 920, 719 #Width: 1280, Height: 720, Channels: 3
    cropped_image = image[y_start:y_end, x_start:x_end]
    processed_file_path = os.path.join(folder, f'processed_{file_name}')
    cv2.imwrite(processed_file_path, cropped_image)
    print(f'Processed {file_name} and saved as {processed_file_path}.')

    '''#create patches
        img_h, img_w, _ = cropped_image.shape
        patch_w, patch_h = (64, 64)
        output_dir = 'patches'
        os.makedirs(output_dir, exist_ok = True)
        patch_id = 0
        # Loop through the image with step size = patch size
        for y in range(0, img_h, patch_h):
            for x in range(0, img_w, patch_w):
 
            # Ensure patch does not exceed image boundaries
            x_end = min(x + patch_w, img_w)
            y_end = min(y + patch_h, img_h)
 
            # Crop the patch
            patch = cropped_image[y:y_end, x:x_end]
 
            # Save the patch
            patch_filename = f"{output_dir}/patch_{patch_id}.png"
            cv2.imwrite(patch_filename, patch)
 
            # Draw a rectangle on the original image (visualization)
            display_image = cropped_image.copy()
            cv2.rectangle(display_image, (x, y), (x_end, y_end), (0, 255, 0), 2)
            patch_id += 1
        # Show the original image with drawn patches
        cv2.imshow("Patches", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''


# == commands ==
#step 1
download_to_local()

#step 2
find_delete_similar_image(folder)

#step 3
for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)
    image_preprosessing(file_path, file_name)