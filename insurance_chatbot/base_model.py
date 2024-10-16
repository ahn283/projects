from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st

# chat prompt template example
prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an AI chatbot having a conversation with a human. Use the following context to understand the human question. Answer only in Korean if I don't ask for an anwer in a specific language.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

llm = ChatOllama(model='llama3.1')

chain = prompt | llm

history = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# streamlit interface
st.title("llama3 ChatBot")

# init chat message
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
for content in st.session_state.chat_history:
    with st.chat_message(content['role']):
        st.markdown(content['message'])


# pass the role "user" to write down the question
if question := st.chat_input("메시지를 입력하세요."):
    
    # display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.chat_history.append(
            {'role':'user', 'message':question}
        )
        
    # replace the invoke() call with stream()
    response = chain_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": "any"}}
    )
    
    
    # display assistant response in chat message container
    with st.chat_message('assistant'):
        st.write(response.content)
        # add assistant response to chat history
        st.session_state.chat_history.append(
            {'role':'assistant', 'message':response.content}
        )