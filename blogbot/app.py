import os
import keyring

# API KEY
OPENAI_API_KEY = keyring.get_password('openai', 'key_for_mac')
ANTHROPIC_API_KEY = keyring.get_password('anthropic', 'key_for_mac')
TAVILY_API_KEY = keyring.get_password('tavily', 'key_for_mac')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# Set up LangSmith observability
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY'] = keyring.get_password('langsmith', 'blogbot')
os.environ['LANGCHAIN_PROJECT'] = "proj-blog-bot"


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

llm_anthropic = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0.5)
llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


from planner import Planner
from researcher import Researcher
from coder import Coder
from writer import Writer
from state import State

state = State()

# nodes
planner = Planner(llm=llm_anthropic)
researcher = Researcher(llm=llm_openai)
coder = Coder(llm=llm_openai)
writer = Writer(llm=llm_anthropic)

from planner import Planner
from researcher import Researcher
from coder import Coder
from writer import Writer

from langgraph.graph import StateGraph, START, END

# Define workflow
builder = StateGraph(State)
builder.add_node("planner", planner.create_node)
builder.add_node("researcher", researcher.create_node)
builder.add_node("coder", coder.create_node)
builder.add_node("writer", writer.create_node)
builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "coder")
builder.add_edge("coder", "writer")
builder.add_edge("writer", END)
graph = builder.compile()

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

app = FastAPI()

class Blog(BaseModel):
    subject: str
    
def generate_blog_stream(blog: Blog):
    """Stream messages from the graph continuously."""
    for s in graph.stream({"messages": [("user", blog.subject)]}):
        yield f"\n--------\n{s}\n\n"  # Server-Sent Events (SSE) format for streaming

@app.post("/write/")
async def write_post(blog: Blog):
    return StreamingResponse(generate_blog_stream(blog), media_type="text/event-stream")
