import streamlit as st
import os

# ------------------------
# IMPORT UTILITIES
# ------------------------
from utils.pdf_reader import extract_text_from_pdf
from utils.text_processing import clean_text, chunk_text
from utils.embeddings import create_faiss_index
from utils.rag_pipeline import answer_question_with_rag
from utils.voice_utils import transcribe_audio_file, transcribe_audio_bytes
from utils.tts_utils import text_to_speech


# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="AI PDF Q&A Assistant (RAG)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# 🌙 DARK THEME
# ------------------------
st.markdown(
    """
<style>
.stApp {
    background-color: #0E1117 !important;
    color: white !important;
}
.block-container {
    background-color: #0E1117 !important;
    color: white !important;
}
.stTextInput>div>div>input {
    background-color: #1E222A !important;
    color: white !important;
}
.stFileUploader {
    background-color: #1E222A !important;
}
.stButton>button {
    background-color: #4A4F57 !important;
    color: white !important;
    border: 1px solid #888;
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------
# SAFE TTS PLAYER
# ------------------------
def play_tts_from_text(text: str):
    audio_result = text_to_speech(text)
    if not audio_result:
        return

    if isinstance(audio_result, (bytes, bytearray)):
        st.audio(audio_result)

    elif isinstance(audio_result, str):
        ext = os.path.splitext(audio_result)[1].lower()
        mime = "audio/wav" if ext == ".wav" else "audio/mp3"
        try:
            with open(audio_result, "rb") as f:
                st.audio(f.read(), format=mime)
        except Exception:
            st.audio(audio_result, format=mime)


# ------------------------
# PAGE TITLE
# ------------------------
st.title("📘 AI PDF Q&A Assistant (RAG)")


# =========================================================
# PDF UPLOAD
# =========================================================

st.header("📄 Upload Your PDF")

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    st.success("PDF uploaded successfully!")

    # -------- TEXT EXTRACTION --------
    with st.spinner("Extracting text..."):
        pdf_text = extract_text_from_pdf(uploaded_pdf)

    # 🚨 SAFETY CHECK 1
    if not pdf_text or pdf_text.strip() == "":
        st.error("❌ No text could be extracted from this PDF. It may be scanned or image-based.")
        st.stop()

    cleaned_text = clean_text(pdf_text)
    chunks = chunk_text(cleaned_text)

    # 🚨 SAFETY CHECK 2
    if not chunks:
        st.error("❌ Text extracted but no chunks were created.")
        st.stop()

    st.session_state.chunks = chunks

    # -------- CREATE EMBEDDINGS --------
    with st.spinner("Creating embeddings... (First time may take ~30 sec)"):
        try:
            faiss_index = create_faiss_index(chunks)
        except Exception as e:
            st.error(f"❌ Error while creating embeddings: {e}")
            st.stop()

    st.session_state.faiss_index = faiss_index

    st.success("✅ PDF processed & ready. Ask anything!")


# =========================================================
# TEXT QUESTION INPUT
# =========================================================

st.header("💬 Ask a Question from the PDF (Text)")

user_question = st.text_input("Type your question here:")

if st.button("Ask 🚀"):
    if "faiss_index" not in st.session_state:
        st.error("Please upload a PDF first!")
    else:
        with st.spinner("Thinking..."):
            short, long = answer_question_with_rag(
                user_question,
                st.session_state.faiss_index,
                st.session_state.chunks,
            )

        st.subheader("🧠 Answer (Short):")
        st.write(short)

        play_tts_from_text(short)

        with st.expander("📘 View More (Detailed Explanation)"):
            st.write(long)


# =========================================================
# VOICE UPLOAD
# =========================================================

st.header("🎤 Ask using Voice (Upload File)")

voice_file = st.file_uploader(
    "Upload voice question (wav/mp3)", type=["wav", "mp3"]
)

if st.button("Ask (Voice Upload) 🔉"):
    if not voice_file:
        st.error("Please upload a voice file first.")
    elif "faiss_index" not in st.session_state:
        st.error("Please upload a PDF first!")
    else:
        with st.spinner("Transcribing your question..."):
            question_text = transcribe_audio_file(voice_file)

        st.write(f"**You asked:** {question_text}")

        with st.spinner("Thinking..."):
            short, long = answer_question_with_rag(
                question_text,
                st.session_state.faiss_index,
                st.session_state.chunks,
            )

        st.subheader("🧠 Answer (Short):")
        st.write(short)

        play_tts_from_text(short)

        with st.expander("📘 View More"):
            st.write(long)


# =========================================================
# LIVE MICROPHONE INPUT
# =========================================================

st.header("🎙 Ask using Live Microphone (RECOMMENDED)")
st.info("Click the mic below → record → then press Ask (Live)")

audio = st.audio_input("🎤 Record your question here")

if st.button("Ask (Live) 🎤🤖"):
    if audio is None:
        st.error("Please record your voice first!")
    elif "faiss_index" not in st.session_state:
        st.error("Please upload a PDF first!")
    else:
        with st.spinner("Transcribing speech..."):
            question_live = transcribe_audio_bytes(audio.getvalue())

        st.write(f"**You said:** {question_live}")

        with st.spinner("Thinking..."):
            short, long = answer_question_with_rag(
                question_live,
                st.session_state.faiss_index,
                st.session_state.chunks,
            )

        st.subheader("🧠 Answer (Short):")
        st.write(short)

        play_tts_from_text(short)

        with st.expander("📘 View More"):
            st.write(long)
