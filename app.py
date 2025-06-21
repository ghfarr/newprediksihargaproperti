import streamlit as st
import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Prediksi Harga Properti", layout="centered")
st.title("🏠 Prediksi Harga Properti")

# Input 7 fitur
grlivarea = st.number_input("Luas Bangunan (m²)", min_value=1)
lotarea = st.number_input("Luas Tanah (m²)", min_value=1)
bedroom = st.number_input("Jumlah Kamar Tidur", min_value=0)
bathroom = st.number_input("Jumlah Kamar Mandi", min_value=0)
garage_area = st.number_input("Ukuran Garasi (m²)", min_value=0)

# Kualitas bangunan
kualitas_opsi = {"Buruk": 3, "Sedang": 5, "Bagus": 8}
kualitas_input = st.selectbox("Kualitas Bangunan", list(kualitas_opsi.keys()))
overallqual = kualitas_opsi[kualitas_input]

# Provinsi Indonesia
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
        lokasi_encoded = label_encoders["Neighborhood"].transform([lokasi_input])[0]
        features = np.array([[grlivarea, lotarea, bedroom, bathroom, overallqual, garage_area, lokasi_encoded]])
        log_pred = model.predict(features)[0]
        harga = np.expm1(log_pred)
        st.success(f"💰 Estimasi Harga Properti: Rp {harga:,.0f}")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
