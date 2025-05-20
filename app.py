import streamlit as st
import openai 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.combine_documents import create_stuff_document_chain
from langchain import ChatPromptTemplate
from langchain import RetrievalChain
from langchain_community.vectorstores import  FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

### load the Groq api

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")
prompt = ChatPromptTemplate.from_template(
    """
   Answer the question based on the provided context only 
   Please provide the most accurate based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("reasearch_paper")  ## Data injestion step 
        st.session_state.docs = st.session_state.loader.load()  ## document loading 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embedding)


user_prompt = st.text_input("Enter your query")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector data base is ready")

import time 

if user_prompt:
    document_chain =  create_stuff_document_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input':user_prompt})
    print(f"response time :{time.process_time()-start}")

    st.write(response['amswer'])
    ## with stremlit  expander 
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------------")







