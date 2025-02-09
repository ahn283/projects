import json
import re

from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from state import FactCheckState
from tools import search, clean_json_output

# Define tools
# Tavily search tool
search_tool = search(max_result=5)

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
        "Summarized issue 1",
        "Summarized issue 2",
        ...
    ]
}
"""

# Class for analysis agent
class Analyzer:
    
    def __init__(self, llm, name:str = "Nicole"):
        self.llm = llm      # The llm model to be used
        self.tools = tools  # The tools to be used
        self.name = name    # The name of an agent
        self.system_prompt = system_prompt  # System prompt
    
    # Create analysis node using create_react_agent api
    def _create_node(self, state) -> FactCheckState:
        # Create ReAct agent
        analysis_agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt)
        result = analysis_agent.invoke(state)
        messages = result["messages"][-1].content
        
        # Get the infromation from the response
        cleaned_messages = clean_json_output(messages)
        
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