from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder upload gambar
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 1. LOAD MODEL MENGGUNAKAN OPENCV DNN (Bypass TFLite library!)
TFLITE_PATH = 'models/cnn_model.tflite'
net = cv2.dnn.readNetFromTensorflow(TFLITE_PATH)

# Label kelas sesuai urutan folder
CLASS_NAMES = ['Kertas (Paper)', 'Batu (Rock)', 'Gunting (Scissors)']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 2. Fungsi Preprocessing Gambar sebelum diprediksi
def predict_image(img_path):
    # Baca gambar pakai OpenCV bawaan
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (150, 150))
    img_array = img_resized.astype(np.float32) / 255.0  # Normalisasi (0-1)
    
    # Masukkan ke OpenCV blob (Batch, Channel, Height, Width)
    blob = cv2.dnn.blobFromImage(img_resized, 1.0/255.0, (150, 150), swapRB=True)
    net.setInput(blob)
    
    # Jalankan prediksi
    predictions = net.forward()
    
    class_idx = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    
    return CLASS_NAMES[class_idx], f"{confidence:.2f}%"

# 3. Route Halaman Utama Web
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Tidak ada bagian file")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Tidak ada file yang dipilih")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result_label, result_confidence = predict_image(filepath)
            
            return render_template(
                'index.html', 
                label=result_label, 
                confidence=result_confidence, 
                img_path=filepath
            )
            
    return render_template('index.html')

def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(debug=True)