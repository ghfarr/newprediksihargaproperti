import streamlit as st
import pickle
import numpy as np

# Load model dan encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Prediksi Harga Properti", layout="centered")
st.title("🏡 Prediksi Harga Properti")
st.write("Masukkan informasi properti Anda:")

# Input fitur sesuai model (5 fitur)
grlivarea = st.number_input("Luas Bangunan (GrLivArea)", min_value=1)
lotarea = st.number_input("Luas Tanah (LotArea)", min_value=1)
overallqual = st.selectbox("Kualitas Bangunan", {"Buruk": 3, "Sedang": 5, "Bagus": 8}.values())
garage_area = st.number_input("Ukuran Garasi (GarageArea)", min_value=0)

# Lokasi (provinsi)
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
provinsi = st.selectbox("Lokasi Properti (Provinsi)", provinsi_list)

# Prediksi harga
if st.button("🔍 Prediksi Harga"):
    try:
        encoded_provinsi = label_encoders["Neighborhood"].transform([provinsi])[0]
        features = np.array([[grlivarea, lotarea, overallqual, garage_area, encoded_provinsi]])
        log_pred = model.predict(features)[0]
        harga = np.expm1(log_pred)
        st.success(f"💰 Estimasi Harga Properti: Rp {harga:,.0f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
