import keyring
import os
import re

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages

from IPython.display import Image, display

# API key
OPENAI_API_KEY = keyring.get_password('openai', 'key_for_mac')
GOOGLE_API_KEY = keyring.get_password('gemini', 'key_for_mac')
ANTHROPIC_API_KEY = keyring.get_password('anthropic', 'key_for_mac')
TAVILY_API_KEY = keyring.get_password('tavily', 'key_for_mac')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# Set up LangSmith observability
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY'] = keyring.get_password('langsmith', 'fact_checker')
os.environ['LANGCHAIN_PROJECT'] = "proj-fact-checker"


                
    
# A function for determining a model
def set_model(llm: str = "gemini", model: str = "gemini-2.0-flash"):
    if llm == "gemini":
        llm = ChatGoogleGenerativeAI(model=model, api_key=GOOGLE_API_KEY)
    elif llm == "openai":
        llm = ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
    elif llm == "anthropic":
        llm = ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY)
    else:
        raise ValueError("No model found")
    return llm

# A function for defining search tool (tavily)
def search(max_result: str = 5):
    search_tool = TavilySearchResults(max_results=max_result)
    return search_tool

# Remove 'json' word from the response of Google Gemini    
def clean_json_output(response: str) -> str:
    """Removes ```json and ``` from LLM output."""
    return re.sub(r"```json\n|\n```", "", response).strip()

def display_graph(graph):
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception as e:
        print("Error: {e}")
        
def invoke_graph(graph, message):
    for s in graph.invoke(
        {
            "messages": [
                ("user", message)
            ]
        }
    ):
        print(s)
        print("---")
        
def stream_graph(graph, message):
    for event in graph.stream(
        {
            "messages": [{"role": "user", "content": message}]
        }
    ):
        for value in event.values():
            # ✅ Ensure 'messages' key exists before accessing it
            if "messages" in value and value["messages"]:
                print("AI:", value["messages"][-1].content)