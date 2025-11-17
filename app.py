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
from utils.ui_components import dark_mode_toggle

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="AI PDF Q&A Assistant (RAG)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📘 AI PDF Q&A Assistant (RAG)")
dark_mode_toggle()

# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    st.success("PDF uploaded successfully!")

    with st.spinner("Extracting text..."):
        pdf_text = extract_text_from_pdf(uploaded_pdf)

    cleaned_text = clean_text(pdf_text)
    chunks = chunk_text(cleaned_text)

    st.session_state.chunks = chunks

    with st.spinner("Creating embeddings..."):
        faiss_index = create_faiss_index(chunks)

    st.session_state.faiss_index = faiss_index

    st.success("PDF processed and ready!")

# =========================================================
# TEXT QUESTION INPUT
# =========================================================

st.header("💬 Chat from PDF")

user_question = st.text_input("Ask a question from the PDF:")

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

        # 🔊 AI SPEAKS ANSWER
        audio_path = text_to_speech(short)
        st.audio(audio_path, format="audio/mp3")

        with st.expander("🔽 View More (Detailed Explanation)"):
            st.write(long)

# =========================================================
# VOICE UPLOAD
# =========================================================

st.header("🎤 Ask using voice (Upload)")

voice_file = st.file_uploader("Upload voice question (wav/mp3)", type=["wav", "mp3"])

if st.button("Ask (Voice Upload) 🔉"):
    if not voice_file:
        st.error("Please upload a voice file first.")
    else:
        with st.spinner("Transcribing..."):
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

        # 🔊 AI SPEAKS ANSWER
        st.audio(text_to_speech(short), format="audio/mp3")

        with st.expander("🔽 View More"):
            st.write(long)

# =========================================================
# LIVE MICROPHONE (BEST OPTION)
# =========================================================

st.header("🎙 Ask using Live Microphone (RECOMMENDED)")
st.info("Click the mic below → record → then press Ask (Live)")

# Streamlit built-in audio recorder
audio = st.audio_input("🎤 Record your question here")

if st.button("Ask (Live) 🎤🤖"):
    if audio is None:
        st.error("Please record your voice first!")
    else:
        # Convert recorded audio to text
        with st.spinner("Transcribing..."):
            question_live = transcribe_audio_bytes(audio.getvalue())

        st.write(f"**You said:** {question_live}")

        # Answer from RAG model
        with st.spinner("Thinking..."):
            short, long = answer_question_with_rag(
                question_live,
                st.session_state.faiss_index,
                st.session_state.chunks,
            )

        st.subheader("🧠 Answer (Short):")
        st.write(short)

        # 🔊 AI SPEAKS THE ANSWER
        st.audio(text_to_speech(short), format="audio/mp3")

        with st.expander("🔽 View More"):
            st.write(long)
