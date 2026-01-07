import streamlit as st
from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver

# -------------------------
# ORIGINAL BACKEND CODE
# -------------------------

API_KEY = os.environ.get('OPEN_AI_API_KEY')

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=API_KEY,
    streaming=True   # ✅ enable streaming
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatNode(state: ChatState):
    messages = state["messages"]

    streamed_text = ""
    placeholder = st.empty()  # UI stream target

    for chunk in model.stream(messages):
        if chunk.content:
            streamed_text += chunk.content
            placeholder.markdown(streamed_text)

    messages.append(SystemMessage(content=streamed_text))
    return {"messages": messages}

graph = StateGraph(ChatState)
checkpoint = MemorySaver()
graph.add_node("chatNode", chatNode)

graph.add_edge(START, "chatNode")
graph.add_edge("chatNode", END)

workflow = graph.compile(checkpointer=checkpoint)

st.set_page_config(page_title="LangGraph ChatBot", layout="centered")
st.title("LangGraph Agentic Chat")

# Initialize persistent state
st.session_state.setdefault("thread_id", "1")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("messages", [])

# Render previous chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
user_input = st.chat_input("Type your query")

if user_input:
    # Add user message to both LangGraph messages and chat_history
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Invoke LangGraph workflow
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initialState = {"messages": st.session_state.messages}

    with st.chat_message("assistant"):
        finalState = workflow.invoke(initialState, config=config)

    # Update persistent state
    st.session_state.messages = finalState["messages"]
    bot_reply = finalState["messages"][-1].content
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    st.markdown(bot_reply)

# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state["chat_history"].clear()
        st.session_state["messages"].clear()
        st.rerun()
