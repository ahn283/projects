# data
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
# embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# vector db
from langchain_community.vectorstores import Chroma
import streamlit as st 
# os
import os
from datetime import datetime

# model info
# llm model selection
models = [
    'gpt-4o',
    'gemini-1.5-pro-latest',
    'llama3.1',
    'mistral'
]

# select an llm model from streamlit sidebar
selected_model = ''

# embedding model for huggingface
embedding_model = 'BAAI/bge-m3'

# file handling functions : upload a file and vectorize it
def process_file(file):
    
    # file upload
    with st.spinner("파일 업로드중..."):
    
        data = file.read()
        _, extension = os.path.splitext(file.name)
        
        # file naming
        current_time = datetime.now()   # 2023-05-25 12:00:00.000
        file_name = f'{current_time.isoformat().replace(":", "_")}{extension}'   # isoformat() : String으로 포맷 변환
        
        # file uploading
        file_path = os.path.join('./data', file_name)
        with open(file_path, 'wb') as f:
            f.write(data)
        
        st.write(f'{file.name} 업로드 완료했습니다.')
        
    # file loading
    if extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif extension == '.text':
        loader = TextLoader(file_path)
    else:
        st.error("지원하지 않는 형식입니다.")      
    
    st.write('loader loaded')
            
    # split 
    docs = loader.load()
    chunk_size = 500
    chunk_overlap = 50
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    data = splitter.split_documents(docs)
    st.write("chunking done!")
    
    # embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
    )
    st.write("embedding model ready...")
    
    #vectordb
    vectorstore_path = './vectorstore'
    vectorstore = Chroma.from_documents(data, embeddings, persist_directory=vectorstore_path)
    st.write("Chroma db ready!")
    vectorstore.persist()
    st.write("Vectorstore created and persisted")