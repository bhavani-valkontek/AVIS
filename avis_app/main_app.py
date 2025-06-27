# app.py
import streamlit as st

# ===============================
# Import individual modules for each tab
# ===============================
from avis_app.scratch_module import scratch_ui
from avis_app.dent_module import dent_ui
from avis_app.rust_module import rust_ui

# ===============================
# Main Streamlit UI for Combined App
# ===============================
st.set_page_config(page_title="🚗 Automatic Vehicle Inspection System", layout="wide")
st.title("🔧 Automatic Vehicle Inspection System")
st.markdown("Use the tabs below to perform scratch, dent, or rust detection.")

# ===============================
# Tabs for three types of inspection
# ===============================
tabs = st.tabs(["🩹 Scratch", "🕳️ Dent", "🛑 Rust"])

# ===============================
# Scratch Tab
# ===============================
with tabs[0]:
    scratch_ui()

# ===============================
# Dent Tab
# ===============================
with tabs[1]:
    dent_ui()

# ===============================
# Rust Tab
# ===============================
with tabs[2]:
    rust_ui()
