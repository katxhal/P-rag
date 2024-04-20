import requests
import streamlit as st
from app.ui import error
from utils.validators import url_validator
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from urllib.parse import urlparse
from openai import OpenAI


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


def process_url(urls, question, model):
    model_local = Ollama(model=model)

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

    # Split the text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
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


def call_to_pplx_pdf(pdf_file, question, api_key):
    print("Calling Perplexity API...")
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

    # Prepare the context
    context = "\n".join(doc_splits)

    payload = {
        "model": "mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    url = "https://api.perplexity.ai/chat/completions"
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        return answer
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        print(error_message)  # Print the error message in the console (for debugging)
        error(error_message, response)
        return None


def call_to_pplx_url(urls, question, api_key):
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
    docs_list = "\n".join([doc.page_content for doc in docs])

    # Split the text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Prepare the context
    context = "\n".join([doc.page_content for doc in doc_splits])

    payload = {
        "model": "mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    url = "https://api.perplexity.ai/chat/completions"
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        return answer
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        print(error_message)  # Print the error message in the console (for debugging)
        error(error_message, response)
        return None
