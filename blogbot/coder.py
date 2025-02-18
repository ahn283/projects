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

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate chart."],
):
    """Use this to execute python code and do match. 
    If you want to see the output of a value,
    you should print it out with `print(....)`. This is visible to user.
    """
    try:
        result = repr.run(code)
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Sucessfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str

tools = [tavily_tool, 
        #  python_repl_tool
    ]

coder_prompt = """
You are a **software developer** responsible for providing well-documented **code examples** for a blog post.

## **🔹 Instructions**
- **Always respond in the same language as the blog topic.**
- Generate **readable, efficient, and well-commented** code snippets.
- Ensure code is **error-free and follows best practices**.
- Provide a brief **explanation** for each snippet.

---

## **🔹 Responsibilities**
### **1. Generate Code Snippets**
- Write **concise and functional** code relevant to the blog topic.
- Ensure proper **syntax and formatting**.

### **2. Explain Code**
- Provide a **step-by-step explanation** of how the code works.
- Highlight **key concepts** and **best practices**.

### **3. Ensure Readability**
- Use **comments** within the code for clarity.

---

## **🔹 Output Format**

# Topic
<Coding Topic>

# Code Snippet
```<Formatted Code Block>```

# Explanation
<Brief explanation of the code>
}
"""


class Coder:
    
    def __init__(self, llm, structure=None):
        
        self.llm = llm
        self.system_prompt = coder_prompt
        self.tools = tools
        self.structure = structure
    
    def create_node(self, state: State):
        
        """A planner defines the blog post structure."""
        agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt)
        
        results = agent.invoke({"messages": state["messages"][-1]})
        
        if self.structure == "hierachical":
            state = {
                "next": "supervisor", 
                "messages": HumanMessage(content=results["messages"][-1].content, name="coder"), 
                "codes": results["messages"][-1].content
            }
        else:
            state = {
                "messages": HumanMessage(content=results["messages"][-1].content, name="coder"), 
                "codes": results["messages"][-1].content
            }
        
        return state