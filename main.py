from app.processing import process_pdf, process_url, call_to_pplx_pdf, call_to_pplx_url
from app.ui import display_messages, sidebar
import streamlit as st
import ollama

# Get the list of locally available models using Ollama API
models_dict = ollama.list()
models = [model_info["model"] for model_info in models_dict["models"]]

# Model selection dropdown
selected_model = st.selectbox("Select Model", models)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title(f"Query using {selected_model} model")
st.write(
    "Enter URLs (one per line) or upload a pdf and a question to query the documents."
)

# Sidebar
use_perplexity, api_key = sidebar()

# User input
question = st.chat_input("Ask a question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Processing..."):
        if st.session_state.pdf_file is not None and question is not None:
            if use_perplexity and api_key:
                answer = call_to_pplx_pdf(st.session_state.pdf_file, question, api_key)
            else:
                answer = process_pdf(
                    st.session_state.pdf_file, question, selected_model
                )
            # answer = process_pdf(st.session_state.pdf_file, question, selected_model)
        elif st.session_state.urls is not None and question is not None:
            if use_perplexity and api_key:
                answer = call_to_pplx_url(st.session_state.urls, question, api_key)
            else:
                answer = process_url(st.session_state.urls, question, selected_model)

            # answer = process_url(st.session_state.urls, question, selected_model)
        else:
            st.error("Please upload a PDF file or enter URLs and a question.")
        if answer is not None:
            if "<|im_end|>" in answer:
                answer = answer.replace("<|im_end|>", "")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            display_messages()
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": "No answer found."}
            )
            display_messages()
