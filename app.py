import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="centered")

# --- 2. LOAD ASSET ---
@st.cache_resource
def load_assets():
    # Pastikan file model_xgb_multiclass.pkl dan scaler.pkl ada di folder yang sama
    model = joblib.load('model/model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("Model atau Scaler tidak ditemukan!")
    st.stop()

# --- 3. MAPPING DATA ---
dict_marital = {"Single": 1, "Married": 2, "Widower": 3, "Divorced": 4, "Facto Union": 5, "Legally Separated": 6}
dict_course = {
    "Biofuel Production Technologies": 33, "Animation and Multimedia Design": 171, "Social Service (Evening)": 9003,
    "Agronomy": 9070, "Communication Design": 9085, "Veterinary Nursing": 9119, "Informatics Engineering": 9130,
    "Equiniculture": 9147, "Management": 9238, "Social Service": 9254, "Tourism": 9500, "Nursing": 9556,
    "Oral Hygiene": 9670, "Advertising and Marketing Management": 9773, "Journalism and Communication": 9853,
    "Basic Education": 9991, "Management (Evening)": 9991
}
target_map = {0: "Dropout", 1: "Graduate"}

# --- 4. UI INPUT ---
st.title("🎓 Prediksi Status Mahasiswa")

with st.form("simple_form"):
    col1, col2 = st.columns(2)
    with col1:
        marital = st.selectbox("Status Pernikahan", list(dict_marital.keys()))
        course = st.selectbox("Program Studi", list(dict_course.keys()))
        gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
        tuition = st.radio("UKT Lunas?", ["Ya", "Tidak"])
        age = st.number_input("Usia Saat Daftar", 17, 70, 20)
    
    with col2:
        s1_approved = st.number_input("Unit Lulus Semester 1", 0, 30, 6)
        s1_grade = st.number_input("IP Semester 1", 0.0, 20.0, 12.0)
        s2_approved = st.number_input("Unit Lulus Semester 2", 0, 30, 6)
        s2_grade = st.number_input("IP Semester 2", 0.0, 20.0, 12.0)
        scholarship = st.radio("Beasiswa?", ["Ya", "Tidak"])

    submit = st.form_submit_button("Cek Prediksi")

# --- 5. PREDIKSI & OUTPUT AKHIR ---
if submit:
    # Buat array 36 kolom (sesuai dataset asli)
    input_features = np.zeros(36)
    
    # Isi fitur penting ke indeks yang benar
    input_dict = {
        0: dict_marital[marital],
        3: dict_course[course],
        16: 1 if tuition == "Ya" else 0,
        17: 1 if gender == "Laki-laki" else 0,
        18: 1 if scholarship == "Ya" else 0,
        19: age,
        24: s1_approved,
        25: s1_grade,
        30: s2_approved,
        31: s2_grade
    }
    
    for idx, value in input_dict.items():
        input_features[idx] = value

    # Nama kolom sesuai dataset agar scaler tidak error
    col_names = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course',
        'Daytime_evening_attendance', 'Previous_qualification', 'Previous_qualification_grade',
        'Nacionality', 'Mothers_qualification', 'Fathers_qualification',
        'Mothers_occupation', 'Fathers_occupation', 'Admission_grade', 'Displaced',
        'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date', 'Gender',
        'Scholarship_holder', 'Age_at_enrollment', 'International',
        'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
        'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
        'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
        'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
        'Unemployment_rate', 'Inflation_rate', 'GDP'
    ]
    
    df_input = pd.DataFrame([input_features], columns=col_names)
    scaled_input = scaler.transform(df_input)
    
    # Prediksi
    prediction = model.predict(scaled_input)[0]
    hasil_akhir = target_map[prediction]

    # Tampilkan hanya hasil akhir
    st.markdown("---")
    if hasil_akhir == "Dropout":
        st.error(f"### Status Prediksi: {hasil_akhir}")
    else:
        st.success(f"### Status Prediksi: {hasil_akhir}")
        
    st.divider()
    
st.sidebar.caption("Sistem Prediksi Jaya Jaya Institut")