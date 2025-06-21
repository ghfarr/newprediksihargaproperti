
import streamlit as st
import pickle
import numpy as np

# Load model dan label encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# UI Config
st.set_page_config(page_title="Prediksi Harga Properti", layout="centered")
st.title("🏠 Prediksi Harga Properti")
st.write("Masukkan detail properti Anda untuk memprediksi estimasi harga terkini.")

# Input Form
grlivarea = st.number_input("Luas Bangunan (m²)", min_value=1, step=1)
lotarea = st.number_input("Luas Tanah (m²)", min_value=1, step=1)

# Dropdown kualitas bangunan
kualitas_opsi = {"Buruk": 3, "Sedang": 5, "Bagus": 8}
kualitas_input = st.selectbox("Kualitas Bangunan", list(kualitas_opsi.keys()))
overallqual = kualitas_opsi[kualitas_input]

# Ukuran garasi
garage_area = st.number_input("Ukuran Garasi (m²)", min_value=0, step=1)

# Lokasi properti (provinsi)
provinsi_list = sorted([
    'Aceh', 'Bali', 'Banten', 'Bengkulu', 'DI Yogyakarta', 'DKI Jakarta',
    'Gorontalo', 'Jambi', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur',
    'Kalimantan Barat', 'Kalimantan Selatan', 'Kalimantan Tengah', 'Kalimantan Timur',
    'Kalimantan Utara', 'Kepulauan Bangka Belitung', 'Kepulauan Riau', 'Lampung',
    'Maluku', 'Maluku Utara', 'Nusa Tenggara Barat', 'Nusa Tenggara Timur',
    'Papua', 'Papua Barat', 'Riau', 'Sulawesi Barat', 'Sulawesi Selatan',
    'Sulawesi Tengah', 'Sulawesi Tenggara', 'Sulawesi Utara', 'Sumatera Barat',
    'Sumatera Selatan', 'Sumatera Utara'
])
lokasi_input = st.selectbox("Lokasi Properti (Provinsi)", provinsi_list)

# Prediksi
if st.button("🔍 Prediksi Harga"):
    try:
        lokasi_encoded = label_encoders['Neighborhood'].transform([lokasi_input])[0]

        input_data = np.array([[grlivarea, lotarea, overallqual, garage_area, lokasi_encoded]])
        log_price_pred = model.predict(input_data)[0]
        harga_prediksi = np.expm1(log_price_pred)  # Kembalikan dari log1p ke harga asli

        st.success(f"💰 Estimasi Harga Properti: Rp {harga_prediksi:,.0f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
