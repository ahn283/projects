import operator

from typing import Annotated, List, Dict
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


# State for fact checkers
class FactCheckState(TypedDict):
    
    # A messages is added after each member finishes
    messages: Annotated[List[BaseMessage], add_messages]
    
    # A list of contention extracted from the query
    contentions: List[str]
    
    # A dictionary mapping each contention to a list of related evidences 
    evidences: Annotated[Dict[str, List[str]], operator.or_]        # Allows updating evidences dynamically
    
    # Use to route work. The supervisor calls a function that will update this every time it makes a decision
    next: str
    
    # Superviosr's instruction to the agent or team
    instruction: str