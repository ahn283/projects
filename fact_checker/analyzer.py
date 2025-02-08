import keyring
import os
import operator
import json
import re

import streamlit as st

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from typing_extensions import TypedDict
from typing import Annotated, List

from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from supervisor import FactCheckState


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

def call_model(llm, query:str):
    response = llm.invoke(query)
    return response

# Define tools
# Tavily search tool
search_tool = TavilySearchResults(max_results=5)

tools = [search_tool]

# System prompt for analysis agent
system_prompt = """
You are a text analysis specialist in a Query Analysis Team.
Your primary role is to extract contentious issues from the input text that require fact-checking.
You should answer the same language as user's input.

# Your Responsibilities:
	1.	Analyze the input text for extracting contention:
Analyze the input text carefully to identify potential areas of controversy or claims requiring verification.
	2.	Extranc specific contentious iuuses:
Extract specific contentious issues, including any claims, statistics, or opinions that might spark debate or require validation.
  	3.	Review Extracted Contentious Issues:
Carefully analyze the issues provided by the query analyst, ensuring you fully understand the context and significance of each issue.
	4.	Summarize Issues Clearly and Concisely:
Create concise summaries that capture the essence of each contentious issue while ensuring clarity and relevance for fact-checking purposes.
	5.	Organize Summaries in a Structured Format:
Present the summarized issues in a clean and structured format that is easy to reference and understand.

# STRICT INSTRUCTIONS:
- **DO NOT use Markdown (` ``` `) in your response.**
- **DO NOT use `json` in the response.**
- **DO NOT wrap the response inside ` ```json ` or any other formatting.**
- **Provide the response as plain text.**

# Output Format: 
{
    "contentious_issues": [
        "[#1] Summarized issue 1",
        "[#2] Summarized issue 2",
        ...
    ]
}
"""

# State for fact checkers
class FactCheckState(TypedDict):
    # A messages is added after each member finishes
    messages: Annotated[List[BaseMessage], add_messages]
    # A list of contention extracted from the query
    contentions: List[str]

# Class for analysis agent
class Analyzer:
    
    def __init__(self, llm, name):
        self.llm = llm      # The llm model to be used
        self.tools = tools  # The tools to be used
        self.name = name    # The name of an agent
        self.system_prompt = system_prompt  # System prompt
    
    # Remove 'json' word from the response of Google Gemini    
    def clean_json_output(self, response: str) -> str:
        """Removes ```json and ``` from LLM output."""
        return re.sub(r"```json\n|\n```", "", response).strip()
    
    # Create analysis node using create_react_agent api
    def _create_node(self, state) -> FactCheckState:
        # Create ReAct agent
        analysis_agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt)
        result = analysis_agent.invoke(state)
        messages = result["messages"][-1].content
        
        # Get the infromation from the response
        cleaned_messages = self.clean_json_output(messages)
        
        try:
            output = json.loads(cleaned_messages)
            summarized_issues = output.get("contentious_issues", [])
            if not summarized_issues:
                raise ValueError("Summarized contentions are empty.")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            summarized_issues = ["Error in summarizing contentions: " + str(e) + str(cleaned_messages)]
        
        return {
            "messages": [HumanMessage(content=messages, name=self.name)],
            "contentions": summarized_issues
        }
    
    # Define the graph
    def create_graph(self):
        workflow = StateGraph(FactCheckState)
        workflow.add_node(self.name, self._create_node)
        workflow.add_edge(START, self.name)
        workflow.add_edge(self.name, END)
        graph = workflow.compile()
        return graph
                
