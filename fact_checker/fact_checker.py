import keyring
import streamlit as st 
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from supervisor import set_model, call_model, FactCheckState, Supervisor
from analyzer import Analyzer

# API key
GOOGLE_API_KEY = keyring.get_password('gemini', 'key_for_mac')

# Streamlit App UI
st.title = ("Fact-Checker")

st.header("🤖Fact-Checker🤖")
st.subheader("AI Team for Fact Checking")

# Available LLM models
models = {
    "OpenAI ChatGPT": "openai",
    "Google Gemini": "gemini",
    "Anthropic Claude": "anthropic"
}

# Model selection dropdown
llm_model = st.selectbox("**Select a large language model:**", models)

# Get the model type
selected_model = models[llm_model]      # Use dictionary lookup instead of list indexing

# Assign model name based on selection
if selected_model == "openai":
    model_name = "gpt-4o-mini"
elif selected_model == "gemini":
    model_name = "gemini-2.0-flash"
elif selected_model == "anthropic": 
    model_name = "claude-3-5-sonnet-latest"
    
# Display the selected model
st.text_input(f"**Model:**", value=model_name)

# Set up LLM
st.write(f"llm: {selected_model}, model: {model_name}")
llm = set_model(selected_model, model_name)

query = st.text_area("Input contentions to be checked..")
if st.button("Verify"):
    name = "Nicole"
    analysis = Analyzer(llm, name)
    app = analysis.create_graph()
    for s in app.stream(
            {
                "messages": [HumanMessage(
                    content=query
                )]
            }
        ):
            if "__end__" not in s:
                with st.chat_message(name):
                    st.write("아래 내용에 대해서 사실 확인이 필요합니다.")
                    st.write()
                    st.write(s[f"{name}"]["messages"][0])
                    st.write("---")