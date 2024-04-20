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
        st.session_state.use_perplexity = st.checkbox("Use Perplexity API")
        if st.session_state.use_perplexity:
            st.session_state.api_key = st.text_input(
                "Enter your Perplexity API Key", type="password"
            )
        else:
            st.session_state.api_key = None
    return st.session_state.use_perplexity, st.session_state.api_key


def error(error_message, response):
    # Check if the error is due to too many tokens for the free tier
    if "exceeds the max limit of 16384 tokens" in response.text:
        st.error(
            "The provided context is too big for the free tier. Please reduce the size and try again."
        )
    else:
        st.error(
            "An error occurred: " + error_message
        )  # Display other errors to the user
