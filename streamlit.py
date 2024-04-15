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

# pdf processing
def process_pdf(pdf_file, question):
    # Read the PDF file
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_text(text)

    # Convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_texts(
        texts=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    # Perform the RAG
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    model_local = Ollama(model="dolphin-mistral")
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# URL processing
def process_url(urls, question):
    model_local = Ollama(model="dolphin-mistral")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    #split the text into chunks
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    
    #convert text chunks into embeddings and store in vector database

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    
    #perform the RAG 
    
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
st.title("Document Query using dolphin-mistral model")
st.write("Enter URLs (one per line) or upload a pdf and a question to query the documents.")
# File uploader
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=50)
# Input field for question
question = st.text_input("Question")


# Button to process input
if st.button('Query Documents'):
    with st.spinner('Processing...'):
        if pdf_file is not None and question is not None:
            answer = process_pdf(pdf_file, question)
        elif urls is not None and question is not None:
            answer = process_url(urls, question)
        else:
            st.error("Please upload a PDF file or enter URLs and a question.")
        if answer is not None:
            # when the answer is returned check if has <|im_end|> in the end of the answer and remove it
            if "<|im_end|>" in answer:
                answer = answer.replace("<|im_end|>", "")
            st.text_area("Answer", value=answer, height=300, disabled=True)
        else:
            st.error("No answer found.")