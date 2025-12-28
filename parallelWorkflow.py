from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
import os

API_KEY = os.environ.get('OPEN_AI_API_KEY')
model = ChatOpenAI(api_key=API_KEY)

class BatsManStates(TypedDict):
    runs: int
    balls: int
    intNumFours: int
    intNumSixes: int
    SR: float
    BPB: float
    BP: float
    Summary: str

# Calculation nodes - return only the computed key
def calcSR(state: BatsManStates):
    sr = (state['runs'] / state['balls']) * 100
    return {'SR': sr}

def calcBPB(state: BatsManStates):
    bpb = state['balls'] / state['intNumFours'] if state['intNumFours'] != 0 else 0
    return {'BPB': bpb}

def calcBP(state: BatsManStates):
    bp = state['balls'] / state['intNumSixes'] if state['intNumSixes'] != 0 else 0
    return {'BP': bp}

# Summary node - reads all keys
def getSummary(state: BatsManStates):
    prompt_str = (
        f"You are a batsman. Your current state is {state['runs']} runs, {state['balls']} balls, "
        f"{state['intNumFours']} fours, {state['intNumSixes']} sixes, "
        f"{state['SR']:.2f} strike rate, {state['BPB']:.2f} balls per four, {state['BP']:.2f} balls per six."
    )
    response = model.invoke(prompt_str)
    return {'Summary': response.content}

# Build the graph
graph = StateGraph(BatsManStates)
graph.add_node("calcSR", calcSR)
graph.add_node("calcBPB", calcBPB)
graph.add_node("calcBP", calcBP)
graph.add_node("getSummary", getSummary)

# Parallel calculation edges
graph.add_edge(START, "calcSR")
graph.add_edge(START, "calcBPB")
graph.add_edge(START, "calcBP")

# All three calculations feed into summary
graph.add_edge("calcSR", "getSummary")
graph.add_edge("calcBPB", "getSummary")
graph.add_edge("calcBP", "getSummary")
graph.add_edge("getSummary", END)

workflow = graph.compile()

# Initial state
initial_state = BatsManStates(
    runs=100, balls=148, intNumFours=50, intNumSixes=90, SR=0, BPB=0, BP=0, Summary=""
)

# Invoke workflow
final_state = workflow.invoke(initial_state)

# Print final summary
print(final_state['Summary'])
  