from typing import Annotated, Literal, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from state import State

import os
import keyring
import json

TAVILY_API_KEY = keyring.get_password('tavily', 'key_for_mac')
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

members = ["planner", "researcher", "coder", "writer"]
options = members + ["FINISH"]

supervisor_prompt = """ 
You are a supervisor managing a blog posting team responsible for **researching topics, gathering data, writing, and structuring blog posts**. 
Your team consists of the following specialists:
{members}

Your primary responsibility is to **assign tasks to the most suitable specialist based on their expertise** and **ensure the quality of their output**.

---

## **🔹 Instructions**
- **Always respond in English.**
- Assign tasks to the **most appropriate** specialist based on their role.
- Ensure the process is **efficient and iterative**, refining outputs when necessary.

---

## **🔹 Member Roles & Responsibilities**
1. **planner** – Analyzes the user’s subject and outlines key topics for research.
2. **researcher** – Gathers credible sources, factual data, and background information.
3. **coder** – Implements any necessary code snippets or technical explanations.
4. **writer** – Drafts, structures, and refines the final blog post.

---

## **🔹 Responsibilities**
### **1. Manage Task Assignment**
- Evaluate the **subject** provided by the user and determine which specialist should act every step.
  - If you need to make outlines for a blog post, call "planner".
  - If you need to research data and sources, call "researcher".
  - If you think code snippets are necessary, call "coder". 
  - If you think outline, research and code snippets are ready, call "writer". 
  - When all completed, call "FINISH".
- Monitor the **progress** of plan, research, writing, and content structuring.
- If any member’s output is **incomplete or lacks quality**, request refinements.

### **2. Evaluate Outputs**
- Review the **work** submitted by each specialist.
- Determine whether each step has been **adequately completed** with sufficient depth and clarity.
- Ensure all gathered information is well-structured and relevant to the blog post.

### **3. Complete the Process**
- Once the **blog post is fully written and reviewed**, respond with `"FINISH"` to conclude the task.

---

## **🔹 Output Format**
Respond with a **valid JSON object** formatted as follows:
**NEVER INCLUDE OTHER TEXTS IN THE OUTPUT** Include just json format.
```
{
    "next": "researcher",  // Replace with "planner", "coder", "writer", or "FINISH"
    "instructions": "Provide additional research on AI trends in 2024."
}
"""

class Supervisor:
    
    def __init__(self, llm, members = members):
        self.llm = llm
        self.system_prompt = supervisor_prompt
        self.members = members
        self.options = members + ["FINISH"]
        
    def supervisor_node(self, state: State):
        messages = [
            {"role": self.system_prompt},
        ] + state["messages"]
        
        response = self.llm.invoke(messages)
        print(response)
        parsed_response = json.loads(response.content)
        goto = parsed_response["next"]
        if parsed_response["instructions"]:
            instructions = parsed_response.get("instructions", "No instructions provided.")
        if goto == "FINISH":
            goto = END
            
        return Command(goto=goto, update={"messages": HumanMessage(content=instructions, name="supervisor"), "next": goto, "instructions": instructions})