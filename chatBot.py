from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict,Literal , Annotated
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from pydantic import BaseModel, Field

API_KEY = os.environ.get('OPEN_AI_API_KEY')

model = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph = StateGraph(ChatState)
def chatNode(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    messages.append(SystemMessage(content=response.content))  # if response is BaseMessage
    return {"messages": messages}


graph.add_node("chatNode",chatNode)

graph.add_edge(START,  "chatNode")
graph.add_edge("chatNode", END)

workflow = graph.compile()
initialState = {"messages" :[HumanMessage(content="Hello how are you")]}
finalState = workflow.invoke(initialState)
print(finalState)

# Extract system messages from the final state
finalState = workflow.invoke(state)

# Get the last message
last_msg = finalState["messages"][-1]

# Make sure it's a SystemMessage
if isinstance(last_msg, SystemMessage):
    print("Bot:", last_msg.content)