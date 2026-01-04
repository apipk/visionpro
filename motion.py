import cv2
import streamlit as st
import os
import pandas as pd
import time
from datetime import datetime
from deepface import DeepFace

# --- 1. KONFIGURASI HALAMAN & UI ---
st.set_page_config(page_title="VisionAI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #3b82f6; color: white; }
    .video-container { border: 4px solid #3b82f6; border-radius: 20px; overflow: hidden; }
    .log-container { background-color: #1e293b; padding: 15px; border-radius: 15px; color: white; height: 400px; overflow-y: auto; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INISIALISASI FOLDER ---
folders = ["faces", "logs"]
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

# --- 3. LOGIKA DATABASE & LOGGING ---
def log_attendance(name, distance):
    file_path = f"logs/attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    now = datetime.now().strftime("%H:%M:%S")
    
    # Hitung akurasi dalam persen (semakin kecil distance, semakin besar persen)
    accuracy = round((1 - distance) * 100, 2)
    
    if not os.path.isfile(file_path):
        df = pd.DataFrame(columns=["Nama", "Waktu", "Akurasi (%)"])
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    if name not in df["Nama"].values:
        new_entry = pd.DataFrame({"Nama": [name], "Waktu": [now], "Akurasi (%)": [accuracy]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(file_path, index=False)
        return True, accuracy
    return False, accuracy

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#3b82f6;'>VisionAI Pro</h2>", unsafe_allow_html=True)
    cam_index = st.number_input("Pilih ID Kamera", 0, 5, 0)
    model_name = st.selectbox("Pilih Model AI", ["VGG-Face", "Facenet", "OpenFace"])
    detector_backend = st.selectbox("Detektor Wajah", ["opencv", "mtcnn", "retinaface"])
    
    st.markdown("---")
    if st.button("🔄 Reset Database Wajah"):
        for file in os.listdir("faces"):
            if file.endswith(".pkl"):
                os.remove(os.path.join("faces", file))
        st.cache_resource.clear()
        st.success("Database di-reset!")

# --- 5. TAMPILAN UTAMA ---
st.markdown("<h1 style='text-align: center; color: white;'>Real-Time Face Recognition Dashboard</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    video_placeholder = st.empty()
    stop_button = st.button("❌ Matikan Sistem", type="secondary")

with col2:
    st.markdown("### 📊 Log Aktivitas")
    log_display = st.empty()
    if 'log_data' not in st.session_state:
        st.session_state.log_data = []

# --- 6. ENGINE KAMERA & AI ---
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    display_frame = frame.copy()
    
    try:
        results = DeepFace.find(
            img_path=frame, 
            db_path="faces", 
            model_name=model_name, 
            detector_backend=detector_backend,
            enforce_detection=False, 
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            res = results[0].iloc[0]
            dist_val = res['distance']
            
            # Batas toleransi (threshold)
            if dist_val < 0.45:
                # Perhitungan Akurasi
                acc_score = (1 - dist_val) * 100
                
                raw_name = os.path.basename(res['identity'])
                identity = os.path.splitext(raw_name)[0].replace('_', ' ').strip()
                
                x, y, w, h = int(res['source_x']), int(res['source_y']), int(res['source_w']), int(res['source_h'])

                # Visualisasi: Hijau jika akurat
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Menampilkan Nama & Persentase Akurasi
                label = f"{identity.upper()} ({acc_score:.1f}%)"
                cv2.putText(display_frame, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Logging Kehadiran
                is_logged, final_acc = log_attendance(identity, dist_val)
                if is_logged:
                    st.session_state.log_data.insert(0, f"✅ {identity.upper()} ({final_acc}%)")
    except Exception:
        pass

    log_content = "<div class='log-container'>" + "".join([f"<p>{l}</p>" for l in st.session_state.log_data[:10]]) + "</div>"
    log_display.markdown(log_content, unsafe_allow_html=True)

    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", width="stretch")
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
st.info("Sistem dimatikan.")