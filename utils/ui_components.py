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
    Good for transparency.
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
# DARK MODE TOGGLE
# ------------------------------------------------------
def dark_mode_toggle():
    """
    Simple UI toggle to enable a dark theme.
    """
    return st.checkbox("🌙 Dark Mode")
