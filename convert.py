import tensorflow as tf

# Muat model .h5 yang berukuran besar
model_path = 'models/cnn_model.h5'
model = tf.keras.models.load_model(model_path)

# Proses konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Simpan model hasil kompresi TFLite
tflite_output_path = 'models/cnn_model.tflite'
with open(tflite_output_path, 'wb') as f:
    f.write(tflite_model)

print("Sukses! Model TFLite berhasil dibuat di:", tflite_output_path)