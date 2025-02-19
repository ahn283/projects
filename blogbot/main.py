import os
import keyring
import json

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
from langchain_core.messages import HumanMessage

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
import asyncio

app = FastAPI()


# Write Blog
## Blog class

class Blog(BaseModel):
    subject: str

## Blog function
async def generate_blog(subject):
    """Generate streamed blog content asynchronously"""
    try:
        for s in graph.stream({"messages": [("user", subject)]}):
            yield f"{s}\n"  # Yield each response chunk
            await asyncio.sleep(0.1)  # Avoid blocking event loop
    except asyncio.CancelledError:
        print("Streaming was cancelled by the client.")  # Handle client disconnects

## Blog post api
@app.post("/write/")
async def write_post(blog: Blog):
    return StreamingResponse(generate_blog(blog.subject), media_type="text/plain")

# Simple chat
## Simple chat class
class Message(BaseModel):
    messages: str

## Simple chat api
@app.post("/chat/")
async def chat(message: Message):
    """Handles incoming chat requests and invokes the LLM."""
    user_message = HumanMessage(content=message.messages)
    response = llm_openai.invoke([user_message])
    return {"Assistant": response.content}  # Extract LLM response content


# Stream test
## Stream test class
class RequestData(BaseModel):
    messages: str

## Stream test function
async def stream_numbers():
    """Generate function to stream numbers 1, 2, and 3 every second."""
    yield json.dumps({"Assistant": "---START---"}) + "\n"
    for i in range(1, 4):
        json_data = json.dumps({"Assistant": str(i)}) + "\n"
        yield json_data
        await asyncio.sleep(1)  # Wait 1 second
    yield json.dumps({"Assistant": "---END---"})

## Stream test api
@app.post("/stream/")
async def stream_response(data: RequestData):
    """Handles the POST request and streams back numbers in JSON format."""
    return StreamingResponse(stream_numbers(), media_type="application/json")