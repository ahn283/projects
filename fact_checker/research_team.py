
import re
import json
from typing import TypedDict, List, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from state import FactCheckState
from tools import set_model, search
from researcher import Researcher


# Tools
search_tool = search(max_result=10)
tools = [search_tool]


# Research team member
members = ["economy", "politics", "social", "culture", "technology", "science", "general"]
names = ["John", "Machiavelli", "Marx", "Rose", "Elon", "Newton", "Rick"]

# Supervisor prompt
supervisor_system = """ 
You are responsible for managing a research team tasked with gathering data and evidence on contentions derived from an analyzer.
Your team consists of the following specialists:
{members}

Your primary role is to **assign contentions** to the most suitable specialist based on their expertise. 
If a contention does not match any specific area, assign it to the **"general" specialist**.

---

## **🔹 Instructions**
- **Always respond in the same language as the contention.**
- Assign contentions to the **most appropriate** specialist based on their expertise.
- If no suitable expert is found, assign the contention to a **general** specialist.

---

## **🔹 Responsibilities**
### **1. Manage Task Assignment**
- Evaluate **each** contention and determine which team member is best suited to research it.
- Monitor the progress of the research process and ensure all contentions are being addressed.
- If a researcher's output is **incomplete or lacks sufficient evidence**, assign them additional tasks.

### **2. Evaluate Research Outputs**
- Review the evidence and sources provided by each specialist.
- Determine whether the contention has been **adequately researched** with strong supporting or opposing evidence.
- If the findings are **sufficient**, update the state with the contention and the gathered evidence.

### **3. Complete the Process**
- Once **all** contentions have been thoroughly researched, respond with `"FINISH"` to conclude the task.

---

## **🔹 Guidelines**
✔️ Assign tasks efficiently to ensure **optimal research flow**.  
✔️ Provide **clear instructions** if a researcher's output needs further refinement.  
✔️ Ensure that the **final results are structured, clear, and actionable** for the fact-checking team.  

---
"""

supervisor_prompt = ChatPromptTemplate.from_template(supervisor_system)

class ResearchTeam:
    
    def __init__(self, llm):
        self.llm = llm
        self.members = members
        self.names = names
        self.tools = tools
        self.supervisor_system = supervisor_system


    def _create_supervisor(self) -> FactCheckState:
        """A supervisor LLM for a research team"""
        
        options = ["FINISH"] + self.members
        
        # Define a function for OpenAI calling
        function_def = {
            "name": "route",
            "description": "Select the next role and provide clear instructions for the selected role to carry out",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": options},
                        ]
                    }
                },
                "required": ["next"],
            }
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.supervisor_system),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next?"
                    " or should we FINISH?, select one of : {options}"
                )
            ]
        ).partial(options=str(", ".join(options)), members=", ".join(members))
        
        return (
            prompt
            | self.llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )
    
    # Add research members
    # Dictionary to store agents dynamically
    def create_graph(self):
        
        supervisor_node = self._create_supervisor()
        
        workflow = StateGraph(FactCheckState)
        workflow.add_node("Supervisor", supervisor_node)
        workflow.add_edge(START, "Supervisor")
        
        research_members = {}
        research_nodes = {}

        for member, name in zip(self.members, self.names):
            research_nodes[member] = Researcher(self.llm, name, member)
            research_members[member] = name
            research_members.update({"FINISH": END})
            workflow.add_node(name, research_nodes[member].research_node)
            workflow.add_edge(name, "Supervisor")
            
        workflow.add_conditional_edges(
            "Supervisor",
            lambda x: x["next"],
            research_members,
        )
        graph = workflow.compile()
        return graph