import keyring
# models
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# ui interface
import streamlit as st
# adding hitory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# data
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
# embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# vector db
from langchain_community.vectorstores import Chroma
# retrieval
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from datetime import datetime
import os


# model selection
models = [
    'gpt-4o',
    'gemini-1.5-pro-latest',
    'llama3.1',
    'mistral'
]

# select model from streamlit sidebar
selected_model = ''


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
            print(file_name)
        
        st.write(f'{file.name} 업로드 완료했습니다.')
        
        if extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif extension == '.text':
            loader = TextLoader(file_path)
        else:
            st.error("지원하지 않는 형식입니다.")
            
    # split 
    docs = loader.load()
    chunk_size = 2000
    chunk_overlap = 200
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    data = splitter.split_documents(docs)
    
    # embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3'
    )
    
    #vectordb
    vectorstore_path = './vectorstore'
    vectorstore = Chroma.from_documents(data, embeddings, persist_directory=vectorstore_path)
    vectorstore.persist()
    print("Vectorstore created and persisted")
    

with st.sidebar:
    selected_model = st.selectbox(
        '어떤 모델로 테스트할 생각이신가요?',
        models
    )  
    # st.write("선택한 모델:", selected_model)
    
    uploaded_file = st.file_uploader("학습할 파일을 선택해주세요. ", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        process_file(uploaded_file)
    

# initialize llm model
if selected_model == models[0]:
    OPENAI_API_KEY = keyring.get_password('openai', 'key_for_windows')
    llm = ChatOpenAI(model=selected_model, api_key=OPENAI_API_KEY)
elif selected_model == models[1]:
    GOOGLE_GEMINI_KEY = keyring.get_password('gemini', 'key_for_windows')
    llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=GOOGLE_GEMINI_KEY)
elif selected_model == models[2]:
    llm = ChatOllama(model='llama3.1')
elif selected_model == models[3]:
    llm = ChatOllama(model='mistral')

# prompt
contextual_system_prompt = """
You are an very helpful AI chatbot having a conversation with a human.\
Be polite.\
Given a chat history and the latest user question \
which might reference context in the chat history, \
formulate a standalone question \
which can be understood without chat history. Do not answer the question, \
just reformulate it if needed and otherwise return it as it is.\
Lastly, answer only in Korean if the user ask for an answer in a specific language.
"""

system_prompt = "You are an AI chatbot having a conversation with a human. Use the following context to understand the human question. Answer only in Korean if I don't ask for an anwer in a specific language."

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human", "{input}")
        
    ]
)

chain = prompt | llm 
history = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key='input',
    history_messages_key='chat_history'
)


# interface
st.title(f"{selected_model} Chatbot")

# initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# display existing chat history    
for content in st.session_state.chat_history:
    with st.chat_message(content['role']):
        st.markdown(content['message'])
        
# pass the role "user" to write down the question
question = st.chat_input("질문을 입력해주세요.")
if question:
    
    # display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.chat_history.append(
            {'role':'user', 'message':question}
        )
        
    # invoke response stream
    # response = chain_with_history.stream(
    #     {"input": question},
    #     config={"configurable": {"session_id": "any"}}
    # )
    
    
    # display assistant's message as it streams and accumulate it
    with st.chat_message('assistant'):  
        
        accumulated_response = ""
        
        with st.spinner("생각 중입니다..."):
            # invoke response stream
            response = chain_with_history.invoke(
                {"input": question},
                config={"configurable": {"session_id": "any"}}
            )   
             
            st.write(response.content)    
            
        
        # save the final answer              
        st.session_state.chat_history.append(
            {'role':'assistant', 'message':response.content}
        )