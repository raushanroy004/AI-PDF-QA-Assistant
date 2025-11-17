import streamlit as st


def pdf_summary_button():
    """
    UI button for generating PDF summary.
    """
    return st.button("📄 Generate PDF Summary")


def ask_question_input():
    """
    UI text input for user question.
    """
    return st.text_input("🔍 Ask a question from the PDF:")


def voice_input_uploader():
    """
    UI uploader for voice file input.
    """
    return st.file_uploader("🎤 Upload voice question (wav/mp3)", type=["wav", "mp3"])


def display_retrieved_chunks(chunks):
    """
    Show retrieved text chunks (sources) for transparency.
    """
    with st.expander("📌 View Source Text Used for Answer"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(chunk)
            st.markdown("---")


def show_chat_history(chat_history):
    """
    Display chat history in a clean message format.
    """
    st.markdown("## 💬 Chat History")
    for entry in chat_history:
        st.markdown(f"**🧑‍💼 You:** {entry['question']}")
        st.markdown(f"**🤖 AI:** {entry['answer']}")
        st.markdown("---")


def download_button(text, filename, label):
    """
    Generic download button for exporting summaries/chat.
    """
    st.download_button(
        label=label,
        data=text,
        file_name=filename,
        mime="text/plain"
    )


def dark_mode_toggle():
    """
    UI dark mode toggle.
    """
    return st.checkbox("🌙 Dark Mode")
