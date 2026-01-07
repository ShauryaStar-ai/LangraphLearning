from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict,Literal , Annotated
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
API_KEY = os.environ.get('OPEN_AI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)