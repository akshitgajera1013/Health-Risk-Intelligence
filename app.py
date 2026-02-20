# ============================================================
# 🏥 Health Risk Intelligence Platform
# Tuned Decision Tree | Hyperparameter Optimized
# Developed by Akshit Gajera
# Enhanced UI with Professional Animations
# ============================================================

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ------------------------------------------------------------
# PAGE CONFIG  (must be the VERY FIRST Streamlit call)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Health Risk Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# PROFESSIONAL ANIMATED THEME
# ------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --primary:      #00d4ff;
    --secondary:    #7b2ff7;
    --accent:       #ff6b35;
    --success:      #00ff9f;
    --danger:       #ff3864;
    --dark-900:     #050b14;
    --glass:        rgba(0,212,255,0.05);
    --glass-border: rgba(0,212,255,0.15);
    --glow:         0 0 20px rgba(0,212,255,0.3);
}

/* ── ANIMATED BACKGROUND ── */
.stApp {
    background: var(--dark-900);
    font-family: 'Rajdhani', sans-serif;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse at 10% 20%, rgba(123,47,247,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(0,212,255,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(255,107,53,0.04) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}
@keyframes bgPulse {
    0%   { opacity: 0.7; }
    100% { opacity: 1.0; }
}

/* ── GRID OVERLAY ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ── MAIN BLOCK ── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 20px;
    padding-bottom: 40px;
    max-width: 1400px;
}

/* ── HERO HEADER ── */
.hero-header {
    text-align: center;
    padding: 50px 20px 30px;
    animation: fadeSlideDown 0.8s ease-out both;
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(24px, 4vw, 52px);
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 30px rgba(0,212,255,0.5));
}
.hero-subtitle {
    font-family: 'Share Tech Mono', monospace;
    color: rgba(0,212,255,0.7);
    font-size: 13px;
    letter-spacing: 4px;
    text-transform: uppercase;
}
.hero-line {
    width: 200px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7b2ff7, transparent);
    margin: 20px auto;
    animation: lineExpand 1s ease-out 0.3s both;
}
@keyframes lineExpand {
    from { width: 0;     opacity: 0; }
    to   { width: 200px; opacity: 1; }
}

/* ── STATUS BAR ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 50px;
    margin-bottom: 24px;
    animation: fadeIn 1s ease-out 0.5s both;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 10px var(--success);
    flex-shrink: 0;
    animation: blink 1.5s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; box-shadow: 0 0 10px var(--success); }
    50%       { opacity: 0.4; box-shadow: 0 0 4px var(--success); }
}
.status-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(0,212,255,0.7);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── GLASS CARD ── */
.glass-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 24px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    animation: cardReveal 0.6s ease-out both;
}
@keyframes cardReveal {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    animation: scanLine 3s linear infinite;
}
@keyframes scanLine {
    0%   { left: -100%; }
    100% { left: 100%; }
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.4);
    box-shadow: var(--glow);
    transform: translateY(-2px);
}
.section-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--primary);
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.8;
}
.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0;
}

/* ── INPUT RESTYLING ── */
div[data-testid="stNumberInput"] > div > div > input,
div[data-testid="stSelectbox"] > div > div {
    background: rgba(0,212,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: #e0f7ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
div[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15) !important;
    outline: none !important;
}
.stSelectbox label,
.stNumberInput label {
    color: rgba(0,212,255,0.85) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* ── PREDICT BUTTON ── */
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%) !important;
    color: #ffffff !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 18px 40px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.3), 0 0 60px rgba(123,47,247,0.2) !important;
}
div.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 50px rgba(0,212,255,0.5), 0 0 80px rgba(123,47,247,0.3) !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── RESULT BOX ── */
.result-box {
    padding: 40px 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
    animation: resultPop 0.5s cubic-bezier(0.175,0.885,0.32,1.275) both;
}
@keyframes resultPop {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}
.result-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(255,255,255,0.05) 60deg, transparent 120deg);
    animation: rotateBg 6s linear infinite;
}
@keyframes rotateBg {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
.result-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(16px, 2.5vw, 28px);
    font-weight: 900;
    letter-spacing: 2px;
    position: relative;
    z-index: 1;
}
.result-confidence {
    font-family: 'Share Tech Mono', monospace;
    font-size: 14px;
    opacity: 0.85;
    margin-top: 10px;
    position: relative;
    z-index: 1;
    letter-spacing: 1px;
}
.result-risk {
    background: linear-gradient(135deg, #1a0010, #4a0020);
    border: 1px solid rgba(255,56,100,0.4);
    box-shadow: 0 0 40px rgba(255,56,100,0.25), inset 0 0 40px rgba(255,56,100,0.05);
}
.result-safe {
    background: linear-gradient(135deg, #001a0f, #004a25);
    border: 1px solid rgba(0,255,159,0.4);
    box-shadow: 0 0 40px rgba(0,255,159,0.25), inset 0 0 40px rgba(0,255,159,0.05);
}

/* ── METRIC CARD ── */
.metric-card {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.3s ease;
}
.metric-card:hover::after { transform: scaleX(1); }
.metric-card:hover {
    background: rgba(0,212,255,0.10);
    transform: translateY(-3px);
    box-shadow: var(--glow);
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 28px;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(0,212,255,0.7);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}
.metric-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    margin-top: 4px;
}

/* ── INPUT SECTION HEADER ── */
.input-section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--primary);
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,212,255,0.15);
    padding-bottom: 10px;
    margin-bottom: 16px;
    opacity: 0.9;
}

/* ── ANALYTICS SECTION HEADER ── */
.analytics-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 28px 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0,212,255,0.12);
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,212,255,0.04) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(0,212,255,0.1) !important;
    padding: 6px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    color: rgba(0,212,255,0.6) !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    transition: all 0.3s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(123,47,247,0.2)) !important;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.2) !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.1) !important;
}
.sidebar-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.sidebar-info-card {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 16px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 14px;
    color: rgba(255,255,255,0.8);
    line-height: 1.9;
}
.sidebar-info-card span { color: var(--primary); font-weight: 600; }

/* ── PROGRESS BAR ── */
div[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    border-radius: 99px !important;
}
div[data-testid="stProgressBar"] {
    background: rgba(0,212,255,0.1) !important;
    border-radius: 99px !important;
}

/* ── DATAFRAME ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-900); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--primary), var(--secondary));
    border-radius: 3px;
}

/* ── FLOATING PARTICLES ── */
.particles-wrap {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.p {
    position: absolute;
    width: 2px;
    height: 2px;
    border-radius: 50%;
    animation: floatUp linear infinite;
}
.p:nth-child(1)  { left:  5%; background:#00d4ff; box-shadow:0 0 6px #00d4ff; animation-duration:12s; animation-delay:0s;   opacity:0.6; }
.p:nth-child(2)  { left: 15%; background:#00d4ff; box-shadow:0 0 6px #00d4ff; animation-duration:18s; animation-delay:2s;   opacity:0.4; }
.p:nth-child(3)  { left: 25%; background:#7b2ff7; box-shadow:0 0 6px #7b2ff7; animation-duration:14s; animation-delay:4s;   opacity:0.7; }
.p:nth-child(4)  { left: 35%; background:#00d4ff; box-shadow:0 0 6px #00d4ff; animation-duration:20s; animation-delay:1s;   opacity:0.3; }
.p:nth-child(5)  { left: 45%; background:#ff6b35; box-shadow:0 0 6px #ff6b35; animation-duration:16s; animation-delay:6s;   opacity:0.5; }
.p:nth-child(6)  { left: 55%; background:#00d4ff; box-shadow:0 0 6px #00d4ff; animation-duration:22s; animation-delay:3s;   opacity:0.4; }
.p:nth-child(7)  { left: 65%; background:#7b2ff7; box-shadow:0 0 6px #7b2ff7; animation-duration:13s; animation-delay:7s;   opacity:0.6; }
.p:nth-child(8)  { left: 75%; background:#00d4ff; box-shadow:0 0 6px #00d4ff; animation-duration:19s; animation-delay:5s;   opacity:0.3; }
.p:nth-child(9)  { left: 85%; background:#00ff9f; box-shadow:0 0 6px #00ff9f; animation-duration:15s; animation-delay:9s;   opacity:0.5; }
.p:nth-child(10) { left: 95%; background:#ff6b35; box-shadow:0 0 6px #ff6b35; animation-duration:17s; animation-delay:0.5s; opacity:0.4; }
@keyframes floatUp {
    0%   { transform: translateY(110vh) scale(0);   opacity: 0; }
    10%  { opacity: 0.8; }
    90%  { opacity: 0.6; }
    100% { transform: translateY(-10vh) scale(1.5); opacity: 0; }
}

/* ── FOOTER ── */
.footer-bar {
    text-align: center;
    padding: 30px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(0,212,255,0.35);
    letter-spacing: 2px;
    text-transform: uppercase;
    border-top: 1px solid rgba(0,212,255,0.08);
    margin-top: 40px;
    position: relative;
    z-index: 1;
}
</style>

<!-- Floating Particles -->
<div class="particles-wrap">
    <div class="p"></div><div class="p"></div><div class="p"></div>
    <div class="p"></div><div class="p"></div><div class="p"></div>
    <div class="p"></div><div class="p"></div><div class="p"></div>
    <div class="p"></div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open("hyper.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

FEATURE_NAMES = [
    "age", "gender", "bmi", "daily_steps", "sleep_hours",
    "water_intake", "calories", "smoker", "alcohol",
    "resting_hr", "systolic_bp", "diastolic_bp",
    "cholesterol", "family_history",
]

# ── Compute pos_index ONCE at module level so every tab can use it ──
POS_INDEX = list(model.classes_).index(1) if 1 in model.classes_ else 0

# ============================================================
# SESSION STATE INIT
# ============================================================
for key in ("prediction", "probabilities", "input_features"):
    if key not in st.session_state:
        st.session_state[key] = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:10px 0 24px;'>
            <div style='font-family:Orbitron,sans-serif; font-size:22px; font-weight:900;
                        background:linear-gradient(135deg,#00d4ff,#7b2ff7);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; letter-spacing:4px;'>HRIP</div>
            <div style='font-family:"Share Tech Mono",monospace; font-size:10px;
                        color:rgba(0,212,255,0.5); letter-spacing:3px; margin-top:4px;'>
                HEALTH RISK INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">&#129504; System Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-info-card">
            <span>Algorithm:</span> Decision Tree<br>
            <span>Optimization:</span> RandomizedSearchCV<br>
            <span>Accuracy:</span> ~75%<br>
            <span>Features:</span> 14 Clinical Inputs<br>
            <span>Task:</span> Binary Classification
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">&#128202; Performance</div>', unsafe_allow_html=True)

    sb_c1, sb_c2 = st.columns(2)
    with sb_c1:
        st.markdown(
            """<div class="metric-card">
                <div class="metric-value">75%</div>
                <div class="metric-label">Accuracy</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with sb_c2:
        st.markdown(
            """<div class="metric-card">
                <div class="metric-value">14</div>
                <div class="metric-label">Features</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(0.75)
    st.caption("Model Accuracy Indicator")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">&#127777; Live Risk Score</div>', unsafe_allow_html=True)

    if st.session_state.probabilities is not None:
        live_risk = round(float(st.session_state.probabilities[POS_INDEX]) * 100, 2)
        rc = "#ff3864" if live_risk >= 70 else ("#ff9f00" if live_risk >= 40 else "#00ff9f")
        st.markdown(
            f"""
            <div style='background:rgba(0,0,0,0.3); border:1px solid {rc}44; border-radius:14px;
                        padding:18px; text-align:center; box-shadow:0 0 20px {rc}22;'>
                <div style='font-family:Orbitron,sans-serif; font-size:36px; font-weight:900;
                            color:{rc}; text-shadow:0 0 20px {rc};'>{live_risk}%</div>
                <div style='font-family:"Share Tech Mono",monospace; font-size:10px;
                            color:rgba(255,255,255,0.5); letter-spacing:2px; margin-top:6px;'>
                    RISK PROBABILITY
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(live_risk / 100)
    else:
        st.markdown(
            """<div style='background:rgba(0,212,255,0.04); border:1px solid rgba(0,212,255,0.1);
                           border-radius:14px; padding:18px; text-align:center;'>
                <div style='font-family:"Share Tech Mono",monospace; font-size:11px;
                            color:rgba(0,212,255,0.5); letter-spacing:2px;'>AWAITING PREDICTION</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(0.0)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️ About the Model"):
        st.markdown(
            """<div style='font-family:Rajdhani,sans-serif; font-size:14px;
                           color:rgba(255,255,255,0.75); line-height:1.9;'>
                • Non-linear decision boundary classification<br>
                • Hyperparameter tuning via RandomizedSearchCV<br>
                • Handles mixed numerical and categorical data<br>
                • Fully interpretable tree splits<br>
                • Validated for medical risk prediction<br>
                • 14-feature clinical input vector
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """<div style='background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(123,47,247,0.15));
                       border:1px solid rgba(0,212,255,0.25); border-radius:14px;
                       padding:14px; text-align:center;'>
            <div style='font-family:Orbitron,sans-serif; font-size:11px; font-weight:700;
                        color:#00d4ff; letter-spacing:2px;'>
                &#128640; ENTERPRISE ML HEALTH SYSTEM
            </div>
            <div style='font-family:"Share Tech Mono",monospace; font-size:10px;
                        color:rgba(255,255,255,0.4); letter-spacing:1px; margin-top:6px;'>
                Developed by Akshit Gajera
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">Health Risk Intelligence</div>
        <div class="hero-subtitle">AI-Based Lifestyle Disease Risk Assessment System</div>
        <div class="hero-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status-bar">
        <div class="status-dot"></div>
        <div class="status-text">
            System Online &nbsp;|&nbsp; Model Loaded &nbsp;|&nbsp;
            14 Features Active &nbsp;|&nbsp; Decision Tree Classifier
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["⚡  PREDICTION ENGINE", "📊  ANALYTICS SUITE", "🌳  MODEL INSIGHTS"]
)

# ============================================================
# TAB 1 — PREDICTION ENGINE
# ============================================================
with tab1:

    st.markdown(
        """<div class="glass-card">
            <div class="section-label">Clinical Input Parameters</div>
            <div class="section-title">Patient Data Entry</div>
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="input-section-header">&#128100; Demographics &amp; Lifestyle</div>',
            unsafe_allow_html=True,
        )
        age         = st.number_input("Age",          min_value=1,   max_value=120,   value=30)
        gender      = st.selectbox("Gender",          ["Male", "Female"])
        bmi         = st.number_input("BMI",          min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        daily_steps = st.number_input("Daily Steps",  min_value=0,   max_value=30000, value=8000, step=100)
        sleep_hours = st.number_input("Sleep Hours",  min_value=0.0, max_value=12.0,  value=7.0,  step=0.5)

    with col2:
        st.markdown(
            '<div class="input-section-header">&#127822; Nutrition &amp; Habits</div>',
            unsafe_allow_html=True,
        )
        water_intake = st.number_input("Water Intake (L)",      min_value=0.0,  max_value=10.0,  value=2.0,  step=0.1)
        calories     = st.number_input("Calories Consumed",     min_value=1000, max_value=6000,  value=2200, step=50)
        smoker       = st.selectbox("Smoker",                   ["No", "Yes"])
        alcohol      = st.selectbox("Alcohol",                  ["No", "Yes"])
        resting_hr   = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=150, value=75)

    with col3:
        st.markdown(
            '<div class="input-section-header">&#128147; Cardiovascular &amp; History</div>',
            unsafe_allow_html=True,
        )
        systolic_bp    = st.number_input("Systolic BP",         min_value=80,  max_value=200, value=120)
        diastolic_bp   = st.number_input("Diastolic BP",        min_value=50,  max_value=130, value=80)
        cholesterol    = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        family_history = st.selectbox("Family History",         ["No", "Yes"])

    # Encode categorical inputs to integers
    gender_enc      = 1 if gender == "Male" else 0
    smoker_enc      = 1 if smoker == "Yes" else 0
    alcohol_enc     = 1 if alcohol == "Yes" else 0
    fam_hist_enc    = 1 if family_history == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("🔍  RUN AI PREDICTION", use_container_width=True)

    if predict_clicked:
        features = np.array([[
            age, gender_enc, bmi, daily_steps,
            sleep_hours, water_intake, calories,
            smoker_enc, alcohol_enc, resting_hr,
            systolic_bp, diastolic_bp,
            cholesterol, fam_hist_enc,
        ]])
        st.session_state.prediction     = model.predict(features)[0]
        st.session_state.probabilities  = model.predict_proba(features)[0]
        st.session_state.input_features = features

    # ── Show results if we have a prediction ──
    if st.session_state.prediction is not None:
        probs            = st.session_state.probabilities
        risk_probability = float(probs[POS_INDEX]) * 100
        confidence       = round(float(np.max(probs)) * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.prediction == 1:
            st.markdown(
                f"""<div class="result-box result-risk">
                    <div class="result-title">&#9888;&#65039; Health Risk Detected</div>
                    <div class="result-confidence">
                        Confidence Score: {confidence}% &nbsp;|&nbsp;
                        Risk Probability: {round(risk_probability, 1)}%
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="result-box result-safe">
                    <div class="result-title">&#9989; No Significant Health Risk</div>
                    <div class="result-confidence">
                        Confidence Score: {confidence}% &nbsp;|&nbsp;
                        Risk Probability: {round(risk_probability, 1)}%
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Gauge ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="analytics-header">&#127919; Risk Scoring Meter</div>', unsafe_allow_html=True)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_probability,
                number={"suffix": "%", "font": {"family": "Orbitron", "size": 36, "color": "#00d4ff"}},
                title={"text": "RISK PROBABILITY", "font": {"family": "Orbitron", "size": 13, "color": "#00d4ff"}},
                delta={"reference": 50, "increasing": {"color": "#ff3864"}, "decreasing": {"color": "#00ff9f"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#00d4ff", "tickfont": {"family": "Share Tech Mono"}},
                    "bar": {"color": "#00d4ff", "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],   "color": "rgba(0,255,159,0.15)"},
                        {"range": [40, 70],  "color": "rgba(255,159,0,0.15)"},
                        {"range": [70, 100], "color": "rgba(255,56,100,0.20)"},
                    ],
                    "threshold": {"line": {"color": "#ff3864", "width": 3}, "thickness": 0.75, "value": 70},
                },
            )
        )
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#00d4ff"},
            height=320,
            margin=dict(l=30, r=30, t=60, b=30),
        )
        st.plotly_chart(gauge, use_container_width=True)

        # ── Quick Metric Cards ──
        mc1, mc2, mc3, mc4 = st.columns(4)
        quick_metrics = [
            ("BMI",         f"{bmi:.1f}",    "Body Mass Index"),
            ("Systolic BP", str(systolic_bp), "mmHg"),
            ("Cholesterol", str(cholesterol), "mg/dL"),
            ("Heart Rate",  str(resting_hr),  "bpm"),
        ]
        for col, (label, val, unit) in zip([mc1, mc2, mc3, mc4], quick_metrics):
            with col:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{label}</div>
                        <div class="metric-unit">{unit}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

# ============================================================
# TAB 2 — ANALYTICS SUITE
# ============================================================
with tab2:

    if st.session_state.input_features is None:
        st.markdown(
            """<div style='text-align:center; padding:80px 20px; font-family:Orbitron,sans-serif;
                           font-size:14px; letter-spacing:3px; color:rgba(0,212,255,0.4);'>
                &#9672; RUN PREDICTION FIRST TO UNLOCK ANALYTICS &#9672;
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        probs = st.session_state.probabilities

        # ── Probability Bar ──
        st.markdown(
            '<div class="analytics-header">&#128202; Prediction Probability Distribution</div>',
            unsafe_allow_html=True,
        )
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=["No Risk", "Risk"],
                y=[float(p) for p in probs],
                marker=dict(
                    color=["rgba(0,255,159,0.7)", "rgba(255,56,100,0.7)"],
                    line=dict(color=["#00ff9f", "#ff3864"], width=2),
                ),
                text=[f"{p * 100:.1f}%" for p in probs],
                textposition="outside",
                textfont=dict(family="Orbitron", size=14, color="white"),
            )
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,212,255,0.03)",
            font=dict(family="Rajdhani", color="#00d4ff"),
            xaxis=dict(gridcolor="rgba(0,212,255,0.08)", color="rgba(0,212,255,0.7)"),
            yaxis=dict(gridcolor="rgba(0,212,255,0.08)", color="rgba(0,212,255,0.7)", tickformat=".0%"),
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Radar Chart ──
        st.markdown(
            '<div class="analytics-header">&#128378; Patient Risk Factor Radar</div>',
            unsafe_allow_html=True,
        )
        base = st.session_state.input_features[0]
        radar_feat_names = [
            "age", "bmi", "daily_steps", "sleep_hours",
            "water_intake", "resting_hr", "systolic_bp", "cholesterol",
        ]
        radar_indices = [FEATURE_NAMES.index(f) for f in radar_feat_names]
        radar_ranges  = [(0,120),(10,50),(0,30000),(0,12),(0,10),(40,150),(80,200),(100,400)]
        norm_vals = []
        for idx, (lo, hi) in zip(radar_indices, radar_ranges):
            v = float(base[idx])
            norm_vals.append(max(0.0, min(100.0, (v - lo) / (hi - lo) * 100)))

        r_closed     = norm_vals + [norm_vals[0]]
        theta_closed = radar_feat_names + [radar_feat_names[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                fill="toself",
                fillcolor="rgba(0,212,255,0.1)",
                line=dict(color="#00d4ff", width=2),
                name="Patient Profile",
            )
        )
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="rgba(0,212,255,0.5)", range=[0, 100]),
                angularaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="rgba(0,212,255,0.7)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Share Tech Mono", color="#00d4ff", size=11),
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Partial Dependence ──
        st.markdown(
            '<div class="analytics-header">&#128200; Interactive Partial Dependence Plot</div>',
            unsafe_allow_html=True,
        )
        selected_feat = st.selectbox("Select Feature to Analyze", FEATURE_NAMES)
        feat_idx      = FEATURE_NAMES.index(selected_feat)
        base_input    = st.session_state.input_features.copy()

        CATEGORICAL = {"gender", "smoker", "alcohol", "family_history"}
        if selected_feat in CATEGORICAL:
            pdp_values = [0, 1]
        else:
            base_val   = float(base_input[0][feat_idx])
            # Guard against zero base value
            lo_val     = base_val * 0.5 if base_val != 0 else -1.0
            hi_val     = base_val * 1.5 if base_val != 0 else  1.0
            pdp_values = np.linspace(lo_val, hi_val, 40).tolist()

        pdp_risk = []
        for val in pdp_values:
            temp_input = base_input.copy()
            temp_input[0][feat_idx] = val
            prob = float(model.predict_proba(temp_input)[0][POS_INDEX])
            pdp_risk.append(prob * 100)

        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=pdp_values,
                y=pdp_risk,
                mode="lines+markers",
                line=dict(color="#00d4ff", width=3, shape="spline"),
                marker=dict(color="#7b2ff7", size=7, line=dict(color="#00d4ff", width=2)),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.06)",
                name="Risk %",
            )
        )
        fig_line.add_hline(
            y=50,
            line=dict(color="#ff3864", width=1.5, dash="dash"),
            annotation_text="50% Threshold",
            annotation_font_color="#ff3864",
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,212,255,0.03)",
            font=dict(family="Rajdhani", color="#00d4ff"),
            xaxis=dict(title=selected_feat, gridcolor="rgba(0,212,255,0.08)", color="rgba(0,212,255,0.7)"),
            yaxis=dict(title="Risk Probability (%)", gridcolor="rgba(0,212,255,0.08)", color="rgba(0,212,255,0.7)"),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_line, use_container_width=True)

# ============================================================
# TAB 3 — MODEL INSIGHTS
# ============================================================
with tab3:

    # ── Decision Tree ──
    st.markdown(
        '<div class="analytics-header">&#127795; Decision Tree Visualization</div>',
        unsafe_allow_html=True,
    )
    fig_tree, ax = plt.subplots(figsize=(20, 10))
    fig_tree.patch.set_facecolor("#050b14")
    ax.set_facecolor("#050b14")
    plot_tree(
        model,
        filled=True,
        feature_names=FEATURE_NAMES,
        class_names=["No Risk", "Risk"],
        ax=ax,
        impurity=False,
        proportion=True,
        rounded=True,
        fontsize=9,
    )
    st.pyplot(fig_tree)
    plt.close(fig_tree)

    # ── Feature Importance ──
    st.markdown(
        '<div class="analytics-header">&#128202; Feature Importance Ranking</div>',
        unsafe_allow_html=True,
    )
    importance_df = (
        pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )
    max_imp = float(importance_df["Importance"].max())
    norm    = (importance_df["Importance"] / max_imp).tolist() if max_imp > 0 else [0.0] * len(importance_df)

    bar_colors = [
        "rgba({},{},{},0.85)".format(int(255*(1-v)), int(100+155*v), int(255*v))
        for v in norm
    ]

    fig_imp = go.Figure(
        go.Bar(
            x=importance_df["Importance"].tolist(),
            y=importance_df["Feature"].tolist(),
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(0,212,255,0.3)", width=1)),
            text=[f"{v:.4f}" for v in importance_df["Importance"]],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=11, color="rgba(0,212,255,0.8)"),
        )
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,212,255,0.03)",
        font=dict(family="Rajdhani", color="#00d4ff"),
        xaxis=dict(title="Importance Score", gridcolor="rgba(0,212,255,0.08)", color="rgba(0,212,255,0.7)"),
        yaxis=dict(gridcolor="rgba(0,212,255,0.08)", color="rgba(0,212,255,0.9)"),
        height=420,
        margin=dict(l=20, r=80, t=20, b=40),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Hyperparameters ──
    st.markdown(
        '<div class="analytics-header">&#9881;&#65039; Best Hyperparameters</div>',
        unsafe_allow_html=True,
    )
    params_df = pd.DataFrame(list(model.get_params().items()), columns=["Parameter", "Value"])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-bar">
        &copy; 2026 &nbsp;|&nbsp; Akshit Gajera &nbsp;|&nbsp;
        Health Risk Intelligence Platform &nbsp;|&nbsp;
        Decision Tree Classifier &nbsp;|&nbsp; Enterprise ML Health System
    </div>
    """,
    unsafe_allow_html=True,
)
