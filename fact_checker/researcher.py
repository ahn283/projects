
import re

from tools import search
from state import FactCheckState
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

# Define tools
search_tool = search(max_result=20)
tools = [search_tool]

system_prompt = """
You are a {specialty} specialist tasked with researching and verifying the factual accuracy of contentions provided by the analyzer. 
Your primary responsibility is to assess the validity of each contention and provide a comprehensive evaluation.

### **🔹 Instructions**
- **Always respond in the same language as the contention.**  
- If the contention is in **Korean**, respond in **Korean**.  
- If the contention is in **English**, respond in **English**.  
- If the contention is in **another language**, match the language accordingly.

### **🔹 Responsibilities**
1. **Use Appropriate Tools & Reliable Sources:**  
   - Verify the factual accuracy of the contention using trusted sources.
   - Ensure all sources are up-to-date and credible.
2. **Assign a Factual Score (0-100):**  
   - **0** = Completely false.  
   - **25** = Almost false but has some truth.  
   - **50** = Partially true.  
   - **75** = Almost true but slightly inaccurate.  
   - **100** = Completely true.  
   - **Always explain the reasoning behind your score, especially if it is not 100 or 0.**  
   
3. **Provide Supporting Evidence:**  
   - Include links to sources that support or refute the contention.  
   - Ensure the sources directly address the claim being evaluated.

### **🔹 Contention to Analyze**
{contention}
"""


class ResearchOutput(BaseModel): 
        
    contention: str = Field(description="contentiona to be checked")
    evidences: list[str] = Field(description="list of evidences and sources for or against the contentions")
    score: int = Field(description="Factual score")

class Researcher:
    
    def __init__(self, llm, name, specialty):
        # Claude model is the best.
        self.llm = llm
        self.tools = tools  # The tools to be used
        self.name = name    # The name of an agent
        self.system_prompt = system_prompt  # System prompt
        self.specialty = specialty
        
    def research_node(self, state: FactCheckState) -> FactCheckState:
        
        """
        Runs research on a contention and returns updated state.
        """
        
        # contention = state["contention"][index]
        
        # Define research_agent
        research_agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt)
        # research_agent = self.llm.bind_tools(tools) | ResearchOutput()

        # Invoke research
        contention = state["messages"][-1].content
        result = research_agent.invoke({"messages": self.system_prompt.format(contention=contention, specialty=self.specialty)})
        
        # Extract results
        messages = result["messages"][-1].content
        
        return {
            "messages": [HumanMessage(content=messages, name=self.name)],
            "evidence": [contention, [messages]],
        }
    
    # Define the graph    
    def create_graph(self):
        workflow = StateGraph(FactCheckState)
        workflow.add_node(self.name, self.research_node)
        workflow.add_edge(START, self.name)
        workflow.add_edge(self.name, END)
        graph = workflow.compile()
        return graph
    
        