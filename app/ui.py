import streamlit as st


def display_messages():
    # Chat UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def sidebar():
    with st.sidebar:
        # File uploader
        st.session_state.pdf_file = st.file_uploader(
            "Upload PDF", type=["pdf"], key="sidebar_uploader"
        )
        # Input fields
        st.session_state.urls = st.text_area(
            "Enter URLs separated by new lines", height=100
        )
        # Toggle for using Perplexity API
        use_perplexity = st.checkbox("Use Perplexity API")
        if use_perplexity:
            api_key = st.text_input("Enter your Perplexity API Key", type="password")
        else:
            api_key = None
    return use_perplexity, api_key
