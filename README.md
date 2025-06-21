# Aplikasi Prediksi Harga Properti 🏠

Aplikasi berbasis Streamlit untuk memprediksi harga properti berdasarkan input pengguna dan model machine learning.

### Fitur Input:
- Luas bangunan & tanah
- Kualitas bangunan
- Ukuran garasi
- Lokasi properti (provinsi di Indonesia)

### Output:
- Estimasi harga properti (dengan model regresi log, dikembalikan ke harga asli menggunakan `np.expm1`).

### Cara Menjalankan:
1. Upload file `app.py`, `model.pkl`, dan `label_encoders.pkl` ke GitHub.
2. Sertakan `requirements.txt`.
3. Deploy ke [Streamlit Cloud](https://streamlit.io/cloud).