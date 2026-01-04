# VisionAI Pro: Real-Time Face Recognition Dashboard

VisionAI Pro adalah aplikasi dasbor pengenalan wajah berbasis web yang menggunakan **Streamlit**, **OpenCV**, dan pustaka **DeepFace**. Aplikasi ini memungkinkan deteksi wajah secara langsung melalui kamera, mencocokkannya dengan database gambar, dan mencatat kehadiran secara otomatis.

## 🚀 Fitur Utama

* **Pengenalan Wajah Real-Time**: Mendeteksi dan mengenali wajah secara langsung dari input kamera.
* **Multi-Model AI**: Mendukung berbagai model seperti VGG-Face, Facenet, dan OpenFace.
* **Fleksibilitas Detektor**: Pilihan backend deteksi wajah menggunakan OpenCV, MTCNN, atau RetinaFace.
* **Sistem Log Otomatis**: Mencatat nama, waktu, dan akurasi deteksi ke dalam file CSV harian di dalam folder `logs/`.
* **Antarmuka Modern**: Dashboard interaktif dengan tema gelap dan log aktivitas real-time.

## 📂 Struktur Proyek

* `motion.py`: Kode utama aplikasi Streamlit.
* `requirements.txt`: Daftar pustaka Python yang diperlukan.
* `faces/`: Folder untuk menyimpan foto referensi wajah (format: `nama.png` atau `nama.jpg`).
* `logs/`: Folder penyimpanan otomatis untuk file log absensi CSV.

## 🛠️ Persyaratan Sistem

Pastikan Anda memiliki Python terinstal, lalu pasang dependensi berikut:
* `streamlit`
* `opencv-python`
* `deepface`
* `tf-keras`
* `pandas`
* `numpy`

## ⚙️ Cara Menjalankan

1.  **Siapkan Database Wajah**:
    Masukkan foto wajah orang-orang yang ingin dikenali ke dalam folder `faces/`. Nama file akan digunakan sebagai identitas (misal: `Ronaldo.png`).

2.  **Instalasi**:
    Buka terminal dan jalankan:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi**:
    Jalankan perintah berikut:
    ```bash
    streamlit run motion.py
    ```

## 📊 Detail Teknis

* **Ambang Batas (Threshold)**: Sistem menggunakan nilai jarak (distance) `< 0.45` untuk mengonfirmasi identitas.
* **Perhitungan Akurasi**: Akurasi dihitung dengan rumus $(1 - \text{distance}) \times 100$.
* **Reset Database**: Terdapat fitur di sidebar untuk menghapus cache model (`.pkl`) jika Anda memperbarui foto di folder `faces/`.

---
*Dikembangkan dengan Python dan DeepFace.*
