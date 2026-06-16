import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# 1. Download & Ekstrak Dataset via URL Resmi yang Baru (Aktif)
url = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
zip_dir = tf.keras.utils.get_file('rps.zip', origin=url, extract=True)

# Lokasi hasil ekstrak dataset
dataset_dir = os.path.join(os.path.dirname(zip_dir), 'rps')

IMG_SIZE = 150
BATCH_SIZE = 32

# 2. Augmentasi Data & Data Loader
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 20% Data untuk Validasi, 80% untuk Training
    horizontal_flip=True,
    rotation_range=20
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 3. Arsitektur Model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 kelas
])

# 4. Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Training Model (5 Epochs)
print("Memulai training model...")
model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

# 6. Simpan Model ke folder 'models'
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/cnn_model.h5')
print("Model berhasil disimpan di 'models/cnn_model.h5'!")