import ollama
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from validators import url as url_validator
from urllib.parse import urlparse

# Get the list of locally available models using Ollama API
models_dict = ollama.list()

# Assuming the structure of models_dict is {'models': [{'model': 'model_name', ...}, ...]}
# Extract model names directly from the dictionary
models = [model_info["model"] for model_info in models_dict["models"]]

# Model selection dropdown
selected_model = st.selectbox("Select Model", models)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# pdf processing
def process_pdf(pdf_file, question, model):
    model_local = Ollama(model=model)
    # Read the PDF file
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_text(text)

    # Convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_texts(
        texts=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
    )
    retriever = vectorstore.as_retriever()

    # Perform the RAG
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)


# URL processing
def process_url(urls, question, model):
    model_local = Ollama(model=model)

    # Convert string of URLs to list
    # Convert string of URLs to list
    urls_list = urls.split("\n")

    # Validate URLs
    valid_urls = []
    for url in urls_list:
        # Check if the URL is valid
        if url_validator(url):
            valid_urls.append(url)
        else:
            # Try adding default scheme if missing
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url_with_scheme = f"http://{url}"
                if url_validator(url_with_scheme):
                    valid_urls.append(url_with_scheme)
                else:
                    print(f"Invalid URL: {url}. Skipping.")
            else:
                print(f"Invalid URL: {url}. Skipping.")

    if not valid_urls:
        st.error("No valid URLs provided. Please enter at least one valid URL.")
        return None

    docs = [WebBaseLoader(url).load() for url in valid_urls]
    docs_list = [item for sublist in docs for item in sublist]

    # split the text into chunks

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # convert text chunks into embeddings and store in vector database

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
    )
    retriever = vectorstore.as_retriever()

    # perform the RAG

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)


# Streamlit UI
st.title(f"Document Query using {selected_model} model")
st.write(
    "Enter URLs (one per line) or upload a pdf and a question to query the documents."
)


# Sidebar
with st.sidebar:
    # File uploader
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="sidebar_uploader")
    # Input fields
    urls = st.text_area("Enter URLs separated by new lines", height=100)


def display_messages():
    # Chat UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# User input
question = st.chat_input("Ask a question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Processing..."):
        if pdf_file is not None and question is not None:
            answer = process_pdf(pdf_file, question, selected_model)
        elif urls is not None and question is not None:
            answer = process_url(urls, question, selected_model)
        else:
            st.error("Please upload a PDF file or enter URLs and a question.")
        if answer is not None:
            # when the answer is returned check if has <|im_end|> in the end of the answer and remove it
            if "<|im_end|>" in answer:
                answer = answer.replace("<|im_end|>", "")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            display_messages()
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": "No answer found."}
            )
            display_messages()
