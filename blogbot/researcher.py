from typing import Annotated, Literal, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

from state import State

import os
import keyring

TAVILY_API_KEY = keyring.get_password('tavily', 'key_for_mac')
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# tools
tavily_tool = TavilySearchResults(max_results=5)

tools = [tavily_tool]

# Researcher
researcher_prompt = """ 
You are a **research specialist** responsible for gathering and verifying **reliable information** for a blog post.

## **🔹 Instructions**
- **Always respond in the same language as the blog topic.**
- Use **credible sources** (e.g., research papers, news articles, official reports).
- **Summarize** key findings concisely.
- Provide **direct links** to sources.

---

## **🔹 Responsibilities**
### **1. Gather Relevant Information**
- Research the topics assigned by the planner.
- Ensure **accuracy and credibility** of sources.

### **2. Summarize Key Insights**
- Extract **essential facts, statistics, or quotes**.
- Summarize findings in **clear and simple language**.

### **3. Provide Sources**
- List **direct links** to the original sources.
"""

class Researcher:
    
    def __init__(self, llm, structure = None):
        
        self.llm = llm
        self.system_prompt = researcher_prompt
        self.tools = tools
        self.structure = structure
    
    
    def create_node(self, state: State):
        
        """A planner defines the blog post structure."""
        
        agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt)
        results = agent.invoke({"messages": state["messages"][-1]})
        
        if self.structure == "hierachical":
            state = {
                "next": "supervisor", 
                "messages": HumanMessage(content=results["messages"][-1].content, name="researcher"),
                "docs": results["messages"][-1].content
            }
        else:
            state = { 
                "messages": HumanMessage(content=results["messages"][-1].content, name="researcher"),
                "docs": results["messages"][-1].content
            }
        
        return state

        
        