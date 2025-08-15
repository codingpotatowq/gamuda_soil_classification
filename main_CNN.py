'''
1. import tensorflow and libraries
2. load image datasets -> normalise with relu and conv2D
3. create model
4. visualise dataset w/ matplotlib
5. build cnn model
6. compile
7. train
8. visualise training output with xy graph
'''

import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import tensorflow as tf
print(f"tensorflow version: {tf.__version__}") 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
import pathlib
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# == Directory == (CHANGE DATASET PATH HERE)
dir_OA = r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Training\OA'
dir_OBOA =  r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Training\OBOA'

directory = [
    dir_OA, dir_OBOA
]

# == Double Checking whether Images are Fetched Correctly ==
total_images =  0
for dir_path in directory:
    data_dir =  Path(dir_path)
    image_count = len(list(data_dir.glob('*.jpg')))
    print(f"Total images found in {dir_path}: {image_count}")
    total_images += image_count

print(f'Total images found across all directories: {total_images}')

if image_count == 0:
    print("Warning: No images found. Check your dataset path and format.")

for dir_path in directory:
    data_dir = Path(dir_path)
    all_files = list(data_dir.glob('*'))
    print(f"Found files in {dir_path} (first 5): {[str(f) for f in all_files[:5]]}")

image_size = 200
batch_size = 10

target_size = (image_size, image_size)
input_shape = (image_size, image_size, 3)



# == Constructing CNN Model ==
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_ds = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.30
    )

train_generator = train_ds.flow_from_directory(
    r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Training',  #CHANGE PATH HERE
    target_size = target_size,
    batch_size = batch_size,
    classes = [
        'OA', 'OBOA' #CHANGE CLASS NAMES HERE
    ],
    class_mode = 'categorical',
    subset = 'training'
    )

validation_generator = train_ds.flow_from_directory(
    r'C:\Users\Tunnel\Documents\Soil_Identification\GCS Training',  #CHANGE PATH HERE
    target_size = target_size,
    batch_size = batch_size,
    classes = [
        'OA', 'OBOA' #CHANGE CLASS NAMES HERE
    ],
    class_mode = 'categorical',
    subset = 'validation'
    )
for image_batch, label_batch in train_generator:
    break


# == Batch inspection for Model Input ==
print("=== BATCH INSPECTION ===")
print(f"Image batch shape: {image_batch.shape}")
print(f"Label batch shape: {label_batch.shape}")
print(f"Number of classes: {label_batch.shape[1]}")
print(f"Batch size: {image_batch.shape[0]}")
print(f"Image dimensions: {image_batch.shape[1:3]}")
print(f"Color channels: {image_batch.shape[3]}")

print(f"\nImage value range: {image_batch.min():.3f} to {image_batch.max():.3f}")
print(f"Image data type: {image_batch.dtype}")

class_names = list(train_generator.class_indices.keys())
predicted_classes = np.argmax(label_batch, axis=1)
print(f"\nClasses in this batch:")
for i, class_idx in enumerate(predicted_classes):
    print(f"  Image {i}: {class_names[class_idx]}")

plt.figure(figsize=(10, 10))
for i in range(min(25, len(image_batch))):
    plt.subplot(5, 5, i+1)
    plt.imshow(image_batch[i])
    class_idx = np.argmax(label_batch[i])
    plt.title(f"{class_names[class_idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
          
print(f'class names: ', train_generator.class_indices)


# == CNN Model ==
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = input_shape), #layers[0]
    tf.keras.layers.MaxPooling2D(2, 2), #layers[1]

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'), #layers[2]
    tf.keras.layers.MaxPooling2D(2, 2), #layers[3]

    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'), #layers[4]
    tf.keras.layers.MaxPooling2D(2, 2), #layers[5]

    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'), #layers[6]
    tf.keras.layers.MaxPooling2D(2, 2), #layers[7]

    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'), #layers[8]
    tf.keras.layers.MaxPooling2D(2, 2), #layers[9]

    tf.keras.layers.Flatten(), #layers[10]
    tf.keras.layers.Dropout(0.5), #layers[11]
    tf.keras.layers.Dense(512, activation = 'relu'), #layers[12 or -3]
    tf.keras.layers.Dense(2, activation = 'softmax') #layers[13]
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

for image_batch, label_batch in validation_generator:
    print("Building model...")
    output = model(image_batch)
    print("Model built successfully!")
    
    print("Model output:")
    print(output)
    break

model.summary()

epochs = 20
history = model.fit(
    train_generator,
    #steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    #validation_steps = validation_generator.samples // batch_size,
    epochs = epochs,
    verbose = 1
)


# == Extracting Features to Plot t-SNE ==
print('Creating feature extractor...')

feature_extractor = tf.keras.Model(
    inputs = model.layers[0].input,
    outputs = model.layers[-3].output
    )

features = []
labels = []
images_analysis = []
batch_count = 0

validation_generator.reset()

for image_batch, label_batch in validation_generator:
    batch_features = feature_extractor(image_batch)
    features.append(batch_features.numpy())
    labels.append(label_batch)
    images_analysis.append(image_batch)

    batch_count += 1
    print(f'Processed batch {batch_count}. ')

    if batch_count >= 50:
        break

features = np.concatenate(features, axis = 0)
labels = np.concatenate(labels, axis = 0)
images_analysis = np.concatenate(images_analysis, axis = 0)

print(f"Feature shape: {features.shape}")
print(f"Label shape: {labels.shape}")

label_indices = np.argmax(labels, axis=1)

print('Computing t-SNE graph...')
#Hyperparameter Settings for TBM Dataset
n_samples = len(features)
perplexity = 15 #min(30, max(5, n_samples // 4)) #(max perplexity = 30, (n_samples x 0.25 = max, but minimum n_samples must be 5))

tsne = TSNE(
    n_components = 2, 
    random_state = 42, 
    perplexity = perplexity,
    #learning_rate = 500,
    max_iter = 250,
    early_exaggeration = 12,
    init = 'pca'
    )

#Hyperparameter Settings for Online Dataset
'''n_samples = len(features)
perplexity = 45 #min(30, max(5, n_samples // 4)) #(max perplexity = 30, (n_samples x 0.25 = max, but minimum n_samples must be 5))

tsne = TSNE(
    n_components = 2, 
    random_state = 42, 
    perplexity = perplexity,
    #learning_rate = 500,
    max_iter = 2500,
    #early_exaggeration = 15,
    init = 'pca'
    )'''


features_2d = tsne.fit_transform(features)

plt.figure(figsize = (12, 8))

colours = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

for i, class_name in enumerate(class_names):
    mask = label_indices == i
    plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                c = [colours[i]], label = class_name, alpha = 0.7, s = 50
                )
        
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Soil Classification Feature Space')
plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
plt.tight_layout()
plt.show()

print('Extraction and Visualisation completed.')


#save model
'''model.save('my_model.h5')
model.save(filepath = 'models/')
from tensorflow.keras.models import Sequential
model.export(export_dir = '.')
converter = tf.lite.TFLiteConverter.from_saved_model('save_model')
tflite_model = converter.convert()
open("soil.tflite", "wb").write(tflite_model)
model.save_weights("model.h5")
'''
