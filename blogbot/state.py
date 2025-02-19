from typing import Literal, List, Optional, Literal, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

class State(TypedDict):
    """Defines the state of the multi-agent blog writing process."""

    # Stores conversation messages (chat history)
    messages: Annotated[List[BaseMessage], add_messages]  

    # Stores collected research materials (articles, references)
    docs: Optional[str]
    
    # Stores generated code snippets
    codes: Optional[str]  
    
    # Stores a generated outline
    outline: Optional[str]
    
    
    # Stores the draft or final version of the blog post
    post: Optional[str]  # ✅ Optional for iterative updates
    
    # Determines which agent should act next (planner, researcher, coder, writer, or FINISH)
    next: Literal["planner", "research", "coder", "writer", "FINISH"]
    
    # Stores specific instructions for the next agent
    instructions: Optional[str]  # ✅ Optional to prevent KeyError when missing