import streamlit as st

# ------------------------------------------------------
# BUTTON: Generate PDF Summary
# ------------------------------------------------------
def pdf_summary_button():
    """
    UI button for generating PDF summary.
    """
    return st.button("📄 Generate PDF Summary")


# ------------------------------------------------------
# TEXT INPUT: Ask Question
# ------------------------------------------------------
def ask_question_input():
    """
    Input box to type a question for the PDF.
    """
    return st.text_input("🔍 Ask a question from the PDF:")


# ------------------------------------------------------
# VOICE FILE UPLOADER
# ------------------------------------------------------
def voice_input_uploader():
    """
    Upload voice files (wav/mp3) for transcription.
    """
    return st.file_uploader("🎤 Upload voice question (wav/mp3)", type=["wav", "mp3"])


# ------------------------------------------------------
# SHOW RETRIEVED CHUNKS
# ------------------------------------------------------
def display_retrieved_chunks(chunks):
    """
    Show the PDF chunks used by the model to answer.
    """
    if not chunks:
        return

    with st.expander("📌 View PDF Source Chunks Used"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"### 📄 Chunk {i+1}")
            st.write(chunk)
            st.markdown("---")


# ------------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------------
def show_chat_history(chat_history):
    """
    Display chat history in a clean message UI.
    """
    if not chat_history:
        return

    st.markdown("## 💬 Chat History")

    for entry in chat_history:
        st.markdown(f"**🧑‍💼 You:** {entry['question']}")
        st.markdown(f"**🤖 AI:** {entry['answer']}")
        st.markdown("---")


# ------------------------------------------------------
# DOWNLOAD BUTTON
# ------------------------------------------------------
def download_button(text, filename, label):
    """
    Generic download button (used for summary/export).
    """
    if not text:
        return

    st.download_button(
        label=label,
        data=text,
        file_name=filename,
        mime="text/plain"
    )


# ------------------------------------------------------
# DARK MODE TOGGLE (FULL WORKING VERSION)
# ------------------------------------------------------
def dark_mode_toggle():
    """
    Fully functional dark mode toggle using custom CSS.
    """

    dark = st.checkbox("🌙 Dark Mode")

    if dark:
        # Apply Dark UI theme
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
            }
            .stDownloadButton>button {
                background-color: #4A4F57 !important;
                color: white !important;
                border: 1px solid #888;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Reset to Light theme
        st.markdown(
            """
            <style>
            .stApp {
                background-color: white !important;
                color: black !important;
            }
            .block-container {
                background-color: white !important;
                color: black !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    return dark
