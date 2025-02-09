import keyring
import streamlit as st 
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from analyzer import Analyzer
from researcher import Researcher
from research_team import ResearchTeam
from tools import set_model

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
    model_name = "claude-3-5-haiku-latest"
    
# Display the selected model
st.text_input(f"**Model:**", value=model_name)

# Set up LLM
st.write(f"llm: {selected_model}, model: {model_name}")
llm = set_model(selected_model, model_name)

query = st.text_area("Input contentions to be checked..")
if st.button("Verify"):
    # # Research Team
    # # Research team member
    # members = ["economy", "politics", "social", "culture", "technology", "science", "general"]
    # names = ["John", "Machiavelli", "Marx", "Rose", "Elon", "Newton", "Rick"]
    # workflow = ResearchTeam(llm=llm)
    
    # # Researcher
    # name = "Nicole"
    # specialty = "politics"
    # workflow = Researcher(llm, name, specialty)
    
    # Analyzer
    name = "Nicole"
    workflow = Analyzer(llm, name)
    
    app = workflow.create_graph()
    for s in app.stream(
            {
                "messages": [HumanMessage(
                    content=query
                )]
            }
        ):
            if "__end__" not in s:
                with st.chat_message(name):
                    st.write("답변드립니다.")
                    
                    # Analyzer
                    st.write(s[f"{name}"]["contentions"])
                    
                    # # Researcher
                    # st.write(s[f"{name}"]["messages"][0].content)
