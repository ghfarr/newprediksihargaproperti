
import streamlit as st
import pickle
import numpy as np

# Load model dan encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Prediksi Harga Properti", page_icon="🏡")
st.title("🏡 Prediksi Harga Properti")
st.caption("Aplikasi cerdas untuk memprediksi harga properti berdasarkan fitur yang Anda input.")

# Input pengguna
luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=10, value=100)
luas_tanah = st.number_input("Luas Tanah (m²)", min_value=10, value=100)
kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=1, value=3)
kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=1, value=2)
ukuran_garasi = st.number_input("Ukuran Garasi (m²)", min_value=5, value=10)

kualitas_bangunan = st.selectbox("Kualitas Bangunan", ["Buruk", "Sedang", "Bagus"])
kualitas_mapping = {"Buruk": 3, "Sedang": 5, "Bagus": 8}
kualitas_input = kualitas_mapping[kualitas_bangunan]

provinsi_list = ['Aceh', 'Banten', 'Bengkulu', 'DI Yogyakarta', 'DKI Jakarta', 'Gorontalo',
                 'Jambi', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Kalimantan Barat',
                 'Kalimantan Selatan', 'Kalimantan Tengah', 'Kalimantan Timur',
                 'Kalimantan Utara', 'Kepulauan Bangka Belitung', 'Kepulauan Riau', 'Lampung',
                 'Maluku', 'Maluku Utara', 'Nusa Tenggara Barat', 'Nusa Tenggara Timur',
                 'Papua', 'Papua Barat', 'Riau', 'Sulawesi Barat', 'Sulawesi Selatan',
                 'Sulawesi Tengah', 'Sulawesi Tenggara', 'Sulawesi Utara', 'Sumatera Barat',
                 'Sumatera Selatan', 'Sumatera Utara']

lokasi = st.selectbox("Lokasi Properti (Provinsi)", provinsi_list)

# Prediksi harga saat tombol ditekan
if st.button("🔍 Prediksi Harga"):
    lokasi_encoded = label_encoders["Neighborhood"].transform([lokasi])[0]
    input_data = np.array([[
        luas_bangunan,
        luas_tanah,
        kamar_tidur,
        kamar_mandi,
        ukuran_garasi,
        kualitas_input,
        lokasi_encoded
    ]])
prediksi = model.predict(input_data)[0]
formatted_price = f"Rp {prediksi:,.0f}".replace(",", ".")
st.success(f"💰 Estimasi Harga Properti: {formatted_price}")

