# app.py

import streamlit as st
import time
import tempfile
import os
from pathlib import Path

# Import your detectors
from text_emotion_detector import TextEmotionDetector
from audio_emotion_detector import predict_audio

# Avatar paths
user_avatar_path = "assets/user.png"
bot_avatar_path = "assets/bot.png"

# ----------------- Helpers -----------------
def load_local_image(path):
    if os.path.exists(path):
        ext = Path(path).suffix.lower()
        if ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]:
            return path
    return None

def format_prob(p):
    try:
        return f"{p:.2f}"
    except:
        return str(p)

# ----------------- UI Setup -----------------
st.set_page_config(page_title="Emotionalverse — ChatUI", layout="wide", initial_sidebar_state="collapsed")

css_path = "styles.css"
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        :root{--bg:#0b1320;--card:#0f1724;--muted:#94a3b8;--accent:#7c3aed;}
        html,body,#root, .reportview-container .main { background: var(--bg) !important; color: #e6eef8; }
        .stButton>button { background: var(--accent); color: white; border-radius:10px; }
        .message-bubble { border-radius:14px; padding:12px; max-width:78%; display:inline-block; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("Emo-verse")
st.markdown("Chat with me to know your emotions! 🧠❤️")

# Load detectors
text_detector = TextEmotionDetector()

# ----------------- Session State -----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# ----------------- Layout -----------------
left, right = st.columns([3, 1])

# ----------------- LEFT PANEL -----------------
with left:
    st.markdown("""<div class="chat-header"><h3>Conversation</h3></div>""", unsafe_allow_html=True)

    for msg in st.session_state.history:
        sender = msg["sender"]
        mtype = msg["type"]
        content = msg["content"]
        result = msg.get("result", {})

        # Avatar
        avatar_path = user_avatar_path if sender == "user" else bot_avatar_path
        avatar_img = load_local_image(avatar_path)
        avatar_html = (
            f'<img class="avatar-img" src="{avatar_img}" />'
            if avatar_img else
            f'<div class="avatar">{"🙂" if sender=="user" else "🤖"}</div>'
        )

        bubble_class = "bubble user-bubble" if sender == "user" else "bubble bot-bubble"

        # --- TEXT MESSAGES ---
        if mtype == "text":
            if sender == "user":
                st.markdown(
                    f"""
                    <div class="message-row user-row">
                        {avatar_html}
                        <div class="{bubble_class} message-bubble">
                            <div class="message-text">{content}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                dom = result.get("dominant_emotion", "unknown")
                conf = format_prob(result.get("confidence", 0.0))
                probs_html = "<br>".join(
                    [f"{k}: {format_prob(v)}" for k, v in result.get("all_emotions", {}).items()]
                )

                st.markdown(
                    f"""
                    <div class="message-row bot-row">
                        {avatar_html}
                        <div class="{bubble_class} message-bubble">
                            <div class="message-text">{content}</div>
                            <div class="message-meta">Emotion: <b>{dom}</b> | Conf: <b>{conf}</b></div>
                            <div class="message-probs">{probs_html}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --- AUDIO MESSAGES ---
        else:
            if sender == "user":
                st.markdown(
                    f"""
                    <div class="message-row user-row">
                        {avatar_html}
                        <div class="{bubble_class} message-bubble">
                            <div class="message-text">🎤 {content}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                dom = result.get("dominant_emotion", "unknown")
                conf = format_prob(result.get("confidence", 0.0))
                probs_html = "<br>".join(
                    [f"{k}: {format_prob(v)}" for k, v in result.get("all_emotions", {}).items()]
                )

                st.markdown(
                    f"""
                    <div class="message-row bot-row">
                        {avatar_html}
                        <div class="{bubble_class} message-bubble">
                            <div class="message-text">🔊 Processed {content}</div>
                            <div class="message-meta">Emotion: <b>{dom}</b> | Conf: <b>{conf}</b></div>
                            <div class="message-probs">{probs_html}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ----------------- RIGHT PANEL -----------------
with right:
    st.markdown("### Input")

    # CLEAR INPUT BEFORE WIDGET RENDERS
    if st.session_state.clear_input:
        st.session_state.input_text = ""
        st.session_state.clear_input = False

    txt = st.text_area("", placeholder="Type a message...", key="input_text", height=150)

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Send"):
            if txt.strip():
                # User msg
                st.session_state.history.append(
                    {"type": "text", "sender": "user", "content": txt}
                )

                # Bot analysis
                with st.spinner("Analyzing..."):
                    res = text_detector.predict(txt)

                st.session_state.history.append(
                    {"type": "text", "sender": "bot", "content": txt, "result": res}
                )

                # FLAG to clear input on next run
                st.session_state.clear_input = True
                st.rerun()

    with cols[1]:
        if st.button("Clear Chat"):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### Upload audio")
    uploaded = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "flac"])

    if uploaded:
        if st.button("Send Audio"):
            suffix = Path(uploaded.name).suffix
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmpf.write(uploaded.getbuffer())
            tmpf.close()

            st.session_state.history.append(
                {"type": "audio", "sender": "user", "content": uploaded.name}
            )

            with st.spinner("Processing audio..."):
                try:
                    audio_res = predict_audio(tmpf.name)
                except Exception as e:
                    audio_res = {"dominant_emotion": "error", "confidence": 0.0, "all_emotions": {"error": str(e)}}

            st.session_state.history.append(
                {"type": "audio", "sender": "bot", "content": uploaded.name, "result": audio_res}
            )

            os.remove(tmpf.name)
            st.rerun()

    st.markdown("---")
    st.markdown("### Settings")
    st.checkbox("Dark Mode", value=True)
    st.checkbox("Show avatars", value=True)
