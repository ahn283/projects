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

tools = [tavily_tool, ]

writer_prompt = """
You are a **blog writer** responsible for crafting an engaging and well-structured blog post.

## **🔹 Instructions**
- **Always write in an engaging and informative tone.**
- Integrate **research findings, code snippets, and planner instructions** into a well-structured article.
- Use **clear headings**, **bullet points**, and **examples** to improve readability.

---

## **🔹 Responsibilities**
### **1. Draft the Blog Post**
- Follow the planner’s **structure and objectives**.
- Ensure the content flows **logically and cohesively**.

### **2. Incorporate Research & Code**
- Seamlessly integrate **research findings** and **code snippets** into the blog.
- Explain technical concepts in **simple language**.

### **3. Ensure Readability & Engagement**
- Use **short paragraphs**, **headings**, and **examples** for clarity.
- Make the content **engaging and easy to understand**.

## Planner outlines
{outline}

## Researches
{docs}

## Code snippet
{codes}

## **🔹 Output Format**
Write the blog post in **Markdown format**. 
Use appropriate **headings (#, ##, ###), lists (-, *), bold text (**bold**), italics (*italics*), and code blocks (```python ... ```) where needed.
"""

class Writer:

    def __init__(self, llm, structure=None):
        
        self.llm = llm
        self.system_prompt = writer_prompt
        self.tools = tools
        self.structure = structure
    
    def create_node(self, state: State):
        
        """A writer defines the blog post structure."""
        outline = state.get("outline", "No outline provided.")
        codes = state.get("codes", "No code snippets available.")
        docs = state.get("docs", "No research documents available.")
        agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt.format(
            # {"outline": outline,
            #  "codes": codes,
            #  "docs": docs}
            outline=outline, codes=codes, docs=docs
        ))
        results = agent.invoke({"messages": state["messages"][-1]})
        
        if self.structure == "hierachical":
            state = {
                "messages": HumanMessage(content=results["messages"][-1].content, name="writer"), 
                "post": results["messages"][-1].content, 
                "next": "supervisor"
            }
        else:
            state = {
                "messages": HumanMessage(content=results["messages"][-1].content, name="writer"), 
                "post": results["messages"][-1].content, 
            }
        
        return state