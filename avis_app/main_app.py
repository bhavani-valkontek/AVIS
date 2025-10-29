# app.py
import streamlit as st
from PIL import Image

# ===============================
# Import modules for each tab
# ===============================
from scratch_module import scratch_ui
from dent_module import dent_ui
from corrosion_module import corrosion_ui
from glass_module import glass_ui
from scratch_module2 import scratch2_ui

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(page_title="AVIS - Auto Vision Inspection System", layout="wide")

# ===============================
# 1️⃣ Logo + Developed By Header
# ===============================
col1, col2 = st.columns([1, 6])
with col1:
    try:
        st.image("logo2.jpg", width=150,)
    except:
        st.warning("⚠️ Logo not found at 'logo2.jpg'")
with col2:
    st.markdown("""
        <div style='padding-top: 10px;'>
            <h2 style='margin-bottom: 5px;'>Valkontek Embedded IOT Services Private Limited</h2>
            <h4>Automatic Vehicle Inspection System
            </h4>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# 2️⃣ App Title & Instructions
# ===============================
# st.markdown("---")
# st.title("🔧 Automatic Vehicle Inspection System")
# st.markdown("Use the tabs below to perform **Scratch**, **Dent**, or **Rust** detection from vehicle images.")

# ===============================
# 3️⃣ Tabs Section
# ===============================
tabs = st.tabs(["Scratch Detection", "Dent Detection", "Corrosion Detection","Glass Detection"])

with tabs[0]:
    scratch_ui()
with tabs[1]:
    scratch2_ui()

with tabs[2]:
    dent_ui()

with tabs[3]:
    corrosion_ui()
with tabs[4]:
    glass_ui()

# ===============================
# 4️⃣ Footer Section
# ===============================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 14px; padding: 10px;'>
        © 2025 <b>Valkontek Embedded IOT Services Pvt Ltd</b>. All rights reserved.
    </div>
""", unsafe_allow_html=True)

