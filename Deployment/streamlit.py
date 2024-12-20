import streamlit as st
import joblib
import numpy as np

# Fungsi normalisasi manual untuk KNN
def manual_normalization(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Fungsi prediksi untuk KNN
def predict_knn(model, input_data):
    # Normalisasi manual input data numerik
    input_data[1] = manual_normalization(input_data[1], 33, 108)  # Berat Badan
    input_data[2] = manual_normalization(input_data[2], 113, 183)  # Tinggi Badan
    input_data[3] = manual_normalization(input_data[3], 50, 120)  # Lingkar Perut
    input_data[4] = manual_normalization(input_data[4], 82, 224)  # Tekanan Darah Sistolik
    input_data[5] = manual_normalization(input_data[5], 50, 137)  # Tekanan Darah Diastolik

    # Konversi ke numpy array dan reshape
    input_array = np.array(input_data).reshape(1, -1)

    # Prediksi
    prediction = model.predict(input_array)
    return "Hipertensi" if prediction[0] == 1 else "Normal (Tidak Hipertensi)"

# Fungsi prediksi untuk Naive Bayes
def predict_nb(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Hipertensi" if prediction[0] == 1 else "Normal (Tidak Hipertensi)"

# Load model
try:
    knn_model = joblib.load('knn_model.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
except Exception as e:
    st.error(f"Error memuat model: {e}")

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="centered")

# Judul aplikasi
st.title("Prediksi Risiko Hipertensi")
st.markdown("Pilih metode prediksi di bawah ini:")

# Tabs untuk model
tabs = st.tabs(["k-Nearest Neighbor", "Naive Bayes Classifier"])

# Tab untuk KNN
with tabs[0]:
    st.subheader("Prediksi Risiko Hipertensi (k-Nearest Neighbor)")

    # Form input pengguna
    with st.form("form_knn"):
        col1, col2, col3 = st.columns(3)

        # Kolom 1
        with col1:
            gender = st.selectbox('Jenis Kelamin', [0, 1], format_func=lambda x: 'Laki-laki' if x == 0 else 'Perempuan')
            weight = st.number_input('Berat Badan (kg)', min_value=0.00, max_value=200.00, step=0.01, format="%.2f")
            height = st.number_input('Tinggi Badan (cm)', min_value=0, max_value=250, step=1)

        # Kolom 2
        with col2:
            waist_circumference = st.number_input('Lingkar Perut (cm)', min_value=0, max_value=200, step=1)
            ap_hi = st.number_input('Tekanan Darah Sistolik (mmHg)', min_value=0, max_value=250, step=1)
            ap_lo = st.number_input('Tekanan Darah Diastolik (mmHg)', min_value=0, max_value=200, step=1)

        # Kolom 3
        with col3:
            smoking = st.radio('Merokok?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            low_physical_activity = st.radio('Kurang Aktivitas Fisik?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            high_sugar = st.radio('Pola Makan Gula Berlebih?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            high_salt = st.radio('Pola Makan Garam Berlebih?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            high_fat = st.radio('Pola Makan Lemak Berlebih?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            low_fruit_veg = st.radio('Kurang Konsumsi Buah & Sayur?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            alcohol = st.radio('Konsumsi Alkohol?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

        # Tombol submit
        submitted_knn = st.form_submit_button("Prediksi Risiko Hipertensi")

        if submitted_knn:
            input_data = [
                gender, weight, height, waist_circumference, ap_hi, ap_lo, smoking,
                low_physical_activity, high_sugar, high_salt, high_fat, low_fruit_veg, alcohol
            ]
            result_knn = predict_knn(knn_model, input_data)
            st.success(f"Hasil Prediksi: {result_knn}")

# Tab untuk Naive Bayes
with tabs[1]:
    st.subheader("Prediksi Risiko Hipertensi (Naive Bayes Classifier)")

    # Form input pengguna
    with st.form("form_nb"):
        col1, col2, col3 = st.columns(3)

        # Kolom 1
        with col1:
            gender = st.selectbox('Jenis Kelamin', [0, 1], format_func=lambda x: 'Laki-laki' if x == 0 else 'Perempuan')
            weight = st.number_input('Berat Badan (kg)', min_value=0.00, max_value=200.00, step=0.01, format="%.2f")
            height = st.number_input('Tinggi Badan (cm)', min_value=0, max_value=250, step=1)

        # Kolom 2
        with col2:
            waist_circumference = st.number_input('Lingkar Perut (cm)', min_value=0, max_value=200, step=1)
            ap_hi = st.number_input('Tekanan Darah Sistolik (mmHg)', min_value=0, max_value=250, step=1)
            ap_lo = st.number_input('Tekanan Darah Diastolik (mmHg)', min_value=0, max_value=200, step=1)

        # Kolom 3
        with col3:
            smoking = st.radio('Merokok?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            low_physical_activity = st.radio('Kurang Aktivitas Fisik?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            high_sugar = st.radio('Pola Makan Gula Berlebih?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            high_salt = st.radio('Pola Makan Garam Berlebih?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            high_fat = st.radio('Pola Makan Lemak Berlebih?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            low_fruit_veg = st.radio('Kurang Konsumsi Buah & Sayur?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
            alcohol = st.radio('Konsumsi Alkohol?', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

        # Tombol submit
        submitted_nb = st.form_submit_button("Prediksi Risiko Hipertensi")

        if submitted_nb:
            input_data = [
                gender, weight, height, waist_circumference, ap_hi, ap_lo, smoking,
                low_physical_activity, high_sugar, high_salt, high_fat, low_fruit_veg, alcohol
            ]
            result_nb = predict_nb(nb_model, input_data)
            st.success(f"Hasil Prediksi: {result_nb}")

# Footer
st.markdown("---")
st.markdown("Aplikasi ini dirancang untuk membantu memprediksi risiko hipertensi berdasarkan data kesehatan.")
st.markdown("By : Adelia Cindy Putri Rahmawati (Statistika Undip 2021)")