# 🤖 Sistem Klasifikasi Citra Gunting-Batu-Kertas (CNN)

Aplikasi web berbasis **Flask** yang mengimplementasikan model **Convolutional Neural Network (CNN)** untuk melakukan klasifikasi citra tangan (Gunting, Batu, atau Kertas) secara *real-time*. Proyek ini dikembangkan untuk memenuhi tugas **Praktikum Kecerdasan Buatan**.

---

## 🎯 Fitur Utama
* **Antarmuka Premium & Dinamis:** Menggunakan Bootstrap 5 dengan tema *Dark Mode* yang responsif.
* **Pratinjau Unggah:** Pengguna dapat melihat gambar pilihan secara lokal sebelum menekan tombol analisis.
* **Efek UX Interaktif:** Menyembunyikan form unggah lama ketika hasil prediksi keluar, menampilkan animasi *loading* saat model sedang memproses citra, dan menyediakan tombol *reset* cepat.
* **Akurasi Tinggi:** Model dilatih dengan arsitektur CNN yang mampu mengenali fitur bentuk tangan secara optimal.

---

## 🛠️ Tech Stack
* **Backend Framework:** Flask (Python)
* **Deep Learning Engine:** TensorFlow / Keras
* **Image Preprocessing:** Pillow, NumPy
* **Frontend UI/UX:** HTML5, Bootstrap 5, FontAwesome, Google Fonts

---

## 📂 Struktur Direktori Proyek
```text
├── models/
│   └── cnn_model.h5         # File Model CNN (Diabaikan oleh Git karena >100MB)
├── static/
│   ├── css/
│   │   └── style.css        # Custom CSS styling
│   └── uploads/             # Folder penyimpanan sementara citra yang diunggah
├── templates/
│   └── index.html           # Template interface web utama (Bootstrap 5)
├── .gitignore               # Konfigurasi pengabaian file Git
├── app.py                   # Driver utama aplikasi web Flask
├── requirements.txt         # Daftar dependencies / library python
└── train.py                 # Script untuk pelatihan model CNN