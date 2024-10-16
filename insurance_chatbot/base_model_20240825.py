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
from langchain_community.embeddings import HuggingFaceEmbeddings
# vector db
from langchain_community.vectorstores import Chroma
# retrieval
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from datetime import datetime
import os
# local libraries
from upload_handler import process_file

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

# retriever
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
vectorstore = Chroma(persist_directory='./vectorstore', embedding_function=embedding_function)
retriever = vectorstore.as_retriever()

contextual_system_prompt = """
You are an AI chatbot having a conversation with a human.\
Given a chat history and the latest user question \
which might reference context in the chat history, \
formulate a standalone question \
which can be understood without chat history. Do not answer the question, \
just reformulate it if needed and otherwise return it as it is.\
"""
contextual_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", contextual_system_prompt
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human", "{input}")
        
    ]
)
system_prompt = (
    """You are an AI chatbot having a conversation with a human.
    Use the following context to understand the human question.
    Answer only in Korean if I don't ask for an anwer in a specific language.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.
    \n\n
    {context}
    """
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

history = StreamlitChatMessageHistory()

system_prompt = "You are an AI chatbot having a conversation with a human. Use the following context to understand the human question. Answer only in Korean if I don't ask for an anwer in a specific language."

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, prompt
)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
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
    
    
    # display assistant's message as it streams and accumulate it
    with st.chat_message('assistant'):  
        
        accumulated_response = ""
        
        with st.spinner("생각 중입니다..."):
            # invoke response stream
            response = chain_with_history.invoke(
                {
                    "input": question,
                    "context": "",
                    "chat_history": st.session_state.chat_history
                },
                config={"configurable": {"session_id": "any"}}
            )   
            st.write(response["answer"])    
            
        
        # save the final answer              
        st.session_state.chat_history.append(
            {'role':'assistant', 'message':response["answer"]}
        )