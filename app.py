from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder upload gambar
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Pastikan folder uploads sudah ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 1. LOAD MODEL MENGGUNAKAN TFLITE INTERPRETER (Update untuk Vercel)
TFLITE_PATH = 'models/cnn_model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

# Mengambil informasi detail input dan output tensor dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label kelas sesuai urutan folder alphabet dataset Rock-Paper-Scissors
CLASS_NAMES = ['Kertas (Paper)', 'Batu (Rock)', 'Gunting (Scissors)']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 2. Fungsi Preprocessing Gambar sebelum diprediksi
def predict_image(img_path):
    # KODE BARU (Gunakan 'preprocessing.image'):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisasi pixel (0-1)
    # Tambah dimensi batch (1, 150, 150, 3) dan paksa tipe data ke float32 sesuai input TFLite
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # LAKUKAN PREDIKSI DENGAN TFLITE INTERPRETER
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    # Ambil hasil output prediksi dari tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    class_idx = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0]) # Persentase keyakinan model
    
    return CLASS_NAMES[class_idx], f"{confidence:.2f}%"

# 3. Route Halaman Utama Web
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Cek apakah ada file gambar yang dikirim
        if 'file' not in request.files:
            return render_template('index.html', error="Tidak ada bagian file")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Tidak ada file yang dipilih")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Panggil fungsi prediksi gambar
            result_label, result_confidence = predict_image(filepath)
            
            # Kirim hasil ke template index.html
            return render_template(
                'index.html', 
                label=result_label, 
                confidence=result_confidence, 
                img_path=filepath
            )
            
    return render_template('index.html')

# Handler WSGI khusus yang diwajibkan oleh Vercel Serverless Function
def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(debug=True)