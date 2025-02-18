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

planner_prompt = """
You are a **blog planner** responsible for structuring and organizing a high-quality blog post. 

## **🔹 Instructions**
- **Always respond in English.**
- Analyze the given topic and define a **clear structure** for the blog post.
- Identify key sections, subtopics, and any necessary research or code that might be needed.
- Provide clear **instructions** for the research, coding, and writing teams.

---

## **🔹 Responsibilities**
### **1. Define Blog Objectives**
- Clearly define the **purpose** and **target audience** of the blog post.
- Identify **key topics** that should be covered.

### **2. Outline Blog Structure**
- Break down the blog into **logical sections** (Introduction, Main Content, Conclusion, etc.).
- Indicate which sections require **research** or **code examples**.

### **3. Assign Tasks**
- Specify which tasks should be assigned to **researchers, coders, and writers**.
- If unsure, assign tasks to the **general** research team.

---

## **🔹 Output**
Respond output as follows:

# Title
<Blog Post Title>

# Objectives
<Main purpose of the blog post>

# Outlines
<Section 1 - Description>
<Section 2 - Description>
<Section 3 - Description>
...
"""

class Planner:
    
    def __init__(self, llm, structure=None):
        
        self.llm = llm
        self.system_prompt = planner_prompt
        self.tools = tools
        self.structure = structure
    
    def create_node(self, state: State):
        
        """A planner defines the blog post structure."""
        
        agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt)
        
        results = agent.invoke({"messages": state["messages"][-1]})
        
        if self.structure == "hierachical":
            state = {
                "messages": HumanMessage(content=results["messages"][-1].content, name="planner"),
                "next": "supervisor",
                "outline": results["messages"][-1].content,
            }
        else:
            state = {
                "messages": HumanMessage(content=results["messages"][-1].content, name="planner"),
                "outline": results["messages"][-1].content,
            }
        
        return state

