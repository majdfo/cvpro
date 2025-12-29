import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import base64
import os
import time

# --- 1. إعدادات الصفحة والتنسيق الاحترافي ---
st.set_page_config(page_title="SafeDrive AI", page_icon="🛡️", layout="wide")

st.markdown("""   <style>
/* تخصيص عنوان "SafeDrive AI" ليكون في أعلى الصفحة */
.stApp .main-header {
    margin-top: 1px; /* تحديد المسافة بين العنوان وبقية العناصر */
    text-align: center; /* توسيط العنوان */
    
}

/* تخصيص العنوان الفرعي */
.stApp .main-header p {
    margin-bottom: 100px; /* تعديل المسافة بين العنوان الفرعي والعنوان الرئيسي */
    text-align: center; /* توسيط النص */
    font-size: 18px; /* حجم الخط */
}

   
    /* 1. تخصيص الشريط الجانبي */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
        padding-top: 20px;  /* إضافة مسافة فوق */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }
    /* 1. تخصيص الشريط الجانبي ليكون باستخدام flexbox */
    [data-testid="stSidebar"] {
        display: flex;
        flex-direction: column;
        align-items: center; /* توسيط المحتوى أفقيًا */
        justify-content: flex-start; /* محاذاة المحتوى من الأعلى */
        background-color: #161b22 !important;
        padding-top: 20px;
    }

    /* 2. تخصيص كلمة "Settings" لتكون في المنتصف */
    [data-testid="stSidebar"] h1 {
        color: white !important; /* اللون الأبيض */
        text-align: center;
        margin-top: 20px; /* إضافة مسافة من الأعلى */
        width: 100%; /* جعل العنوان يحتل كامل العرض */
        text-align: center; /* توسيط النص */
    }
 


    /* 3. تخصيص الصورة داخل الشريط الجانبي لتكون في المنتصف */
    [data-testid="stSidebar"] img {
        display: block;
        margin: 0 auto 20px; /* توسيط الصورة */
        width: 50%; /* يمكن تعديل العرض حسب الرغبة */
    }
        /* تخصيص الـ radio buttons و checkboxes */
    div[data-testid="stRadio"] label, 
    div[data-testid="stCheckbox"] label {
        color: white !important; /* لون النص أبيض */
    }
 
    /* تخصيص النصوص في الشريط الجانبي لتكون باللون الأبيض */
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: white !important; /* اللون الأبيض للنصوص */
    }

    /* تغيير لون كلمة "Settings" داخل الشريط الجانبي */
    [data-testid="stSidebar"] h1 {
        color: white !important; /* تغيير لون العنوان إلى الأبيض */
        text-align: center;
    }

    /* تخصيص الـ radio buttons و checkboxes */
    div[data-testid="stRadio"] label, 
    div[data-testid="stCheckbox"] label {
        color: white !important; /* لون النص أبيض */
    }

    /* تخصيص الـ radio buttons والـ checkboxes */
    div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        margin-bottom: 8px !important;
        min-height: 55px !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.3s ease;
    }

    div[data-testid="stRadio"] label:hover, div[data-testid="stCheckbox"] label:hover {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-color: #3b82f6 !important;
    }

    /* تخصيص السلايدر ليظهر نصه أبيض */
    .stSlider label {
        color: white !important;
    }
    
    /* تخصيص السلايدر ليظهر نصه أبيض */
    .stSlider label {
        color: white !important;
    }

    /* 4. تخصيص الـ radio buttons والـ checkboxes */
    div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label {
        background-color: #21262d !important; /* لون خلفية داكن */
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 12px 15px !important;
        margin-bottom: 8px !important;
        width: 100% !important;
        min-height: 50px !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.3s ease !important;
        color: #ffffff !important;
    }

    /* تأثير عند تمرير الماوس على الـ radio و checkbox */
    div[data-testid="stRadio"] label:hover, 
    div[data-testid="stCheckbox"] label:hover {
        border-color: #58a6ff !important;
        background-color: #30363d !important;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.2) !important;
    }

    /* إخفاء دوائر الراديو التقليدية */
    div[data-testid="stRadio"] input[type="radio"] {
        display: none;
    }

    /* 5. تخصيص السلايدر ليظهر نصه أبيض */
    .stSlider label {
        color: white !important;
    }

    /* 6. تخصيص الأزرار ليكون لها تأثير عند التمرير */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 600;
        transition: transform 0.2s ease;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }

    /* تحسين نتائج الكروت */
    .result-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>""", unsafe_allow_html=True)


# ... (باقي الكود الخاص بـ Columns وتوزيع الـ Input Source)
# --- 2. تحميل الموديلات والوظائف ---
@st.cache_resource
def load_all_models():
    v8 = YOLO("best.pt")
    v11 = YOLO("best11.pt")
    return v8, v11


model_v8, model_v11 = load_all_models()


def play_alert_sound():
    if os.path.exists("alert_sound.mp3"):
        with open("alert_sound.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>',
                        unsafe_allow_html=True)


def analyze_image(img, model, model_name, container):
    start_time = time.time()
    results = model.predict(img, conf=confidence)[0]
    elapsed_time = time.time() - start_time

    container.image(results.plot(), use_container_width=True)
    labels = [model.names[int(box.cls)] for box in results.boxes]
    distractions = [l for l in labels if l in ['PhoneUse', 'Smoking']]

    if distractions:
        container.error(f"🚨 {model_name}: Distraction ({', '.join(set(distractions))})")
        play_alert_sound()
    elif 'Seatbelt' in labels:
        container.success(f"✅ {model_name}: Safe")
    else:
        container.info(f"🟢 {model_name}: Normal")
    container.caption(f"⏱️ Speed: {elapsed_time:.3f}s")


# --- 3. الهيكل (Sidebar & Dashboard) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2555/2555013.png", width=80)
    st.title("Settings")
    st.markdown("---")

    input_source = st.radio("Input Source:", ["Upload Image 🖼️", "Take a Photo 📸", "Live Stream 🎥"])
    engine_choice = st.radio("Detection Engine:", ["YOLOv8", "YOLOv11", "Both (Comparison Mode)"])

# الواجهة الرئيسية
st.markdown('<div class="main-header"><h1>🛡️ SafeDrive AI</h1><p>Advanced Driver Monitoring System</p></div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### 🛠️ Control Panel")
    with st.expander("Model Hyperparameters", expanded=True):
        confidence = st.slider("Confidence", 0.0, 1.0, 0.45)
        st.info("Higher confidence reduces false alarms.")

with col2:
    st.markdown(f"###  {input_source}")

    data_file = None
    if input_source == "Upload Image 🖼️":
        data_file = st.file_uploader("Select image file...", type=['jpg', 'jpeg', 'png'])
    elif input_source == "Take a Photo 📸":
        data_file = st.camera_input("Capture driver snapshot")

    # معالجة الصور (رفع أو التقاط)
    if data_file and input_source != "Live Stream 🎥":
        img_array = np.array(Image.open(data_file))
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if engine_choice == "Both (Comparison Mode)":
            sub_col1, sub_col2 = st.columns(2)
            analyze_image(img_array, model_v8, "YOLOv8", sub_col1)
            analyze_image(img_array, model_v11, "YOLOv11", sub_col2)
        else:
            active_model = model_v8 if engine_choice == "YOLOv8" else model_v11
            analyze_image(img_array, active_model, engine_choice, st)
        st.markdown('</div>', unsafe_allow_html=True)

    # معالجة البث المباشر
    elif input_source == "Live Stream 🎥":
        class StreamProcessor(VideoTransformerBase):
            def __init__(self, engine, m8, m11):
                self.engine, self.m8, self.m11 = engine, m8, m11

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                if self.engine == "Both (Comparison Mode)":
                    return np.hstack((self.m8.predict(img)[0].plot(), self.m11.predict(img)[0].plot()))
                target = self.m8 if self.engine == "YOLOv8" else self.m11
                return target.predict(img)[0].plot()


        webrtc_streamer(
            key="stream",
            video_processor_factory=lambda: StreamProcessor(engine_choice, model_v8, model_v11),
            media_stream_constraints={"video": True, "audio": False},
        )