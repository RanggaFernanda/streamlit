
import tensorflow as tf
import streamlit as st
import numpy as np
import pickle
import base64
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============== BACKGROUND IMAGE ==============
with open("timnas.png", "rb") as img:
    bg_base64 = base64.b64encode(img.read()).decode()

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: 
            linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)),
            url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============== LOAD CSS ==============
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
# ============== LOAD CSS ==============
# ============== LOGO (LEFT TOP) ==============
def load_image_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

logo_base64 = load_image_base64("logo.png")

st.markdown(
    f"""
    <div class="main-title">
        <img src="data:image/png;base64,{logo_base64}" style="width:80px;">
    </div>
    """,
    unsafe_allow_html=True
)


# ============== LOAD MODEL ==============
model = tf.keras.models.load_model(
    "sentiment_analysis_model.keras",
    compile=False
)


# ============== LOAD TOKENIZER ==============
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

MAXLEN = 100  # Sesuaikan dengan training

st.markdown("<div class='main-title'>Performa Timnas Indonesia Kualifikasi Piala Dunia 2026</div>", unsafe_allow_html=True)
# ============== HEADER ==============
st.markdown("<div class='main-title'>🔍 Analisis Sentimen</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Model BiLSTM untuk mendeteksi sentimen positif atau negatif</div>", unsafe_allow_html=True)

#st.markdown("<div class='input-card'>", unsafe_allow_html=True)

text = st.text_area("Masukkan kalimat:", height=150)

st.markdown(
    """
    <style>
    div[data-testid="stTextArea"] label {
        color: white !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============== PREDIKSI ==============
if st.button("🚀 Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu!")

    else:

        # ===== CEK RELEVANSI DENGAN TOKENIZER =====
        sequence_check = tokenizer.texts_to_sequences([text])[0]

        known_words = len(sequence_check)

        # Threshold bisa disesuaikan
        if known_words < 3:

            st.markdown(
                """
                <div class='result-box' style='background:#616161;'>
                    TIDAK RELEVAN
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                    margin-top:15px;
                    font-size:18px;
                    text-align:center;
                    color:white;">
                    Teks tidak berkaitan dengan performa Timnas Indonesia
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            sequence = tokenizer.texts_to_sequences([text])

            padded = pad_sequences(
                sequence,
                maxlen=MAXLEN
            )

            prediction = model.predict(
                padded,
                verbose=0
            )[0][0]

            prob_pos = float(prediction)
            prob_neg = float(1 - prediction)

            if prediction >= 0.5:

                label = "POSITIF"
                color = "background:#00c853;"
                nilai = prob_pos
                keterangan = "Nilai Prediksi Positif"

            else:

                label = "NEGATIF"
                color = "background:#d50000;"
                nilai = prob_neg
                keterangan = "Nilai Prediksi Negatif"

            st.markdown(
                f"""
                <div class='result-box' style='{color}'>
                    {label}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                    margin-top:15px;
                    font-size:18px;
                    text-align:center;
                    color:white;">
                    <b>{keterangan}:</b> {nilai:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )


  
