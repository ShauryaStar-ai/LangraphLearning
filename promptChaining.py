from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
import os

# Load your OpenAI API key from environment
API_KEY = os.environ.get('OPEN_AI_API_KEY')
model = ChatOpenAI(api_key=API_KEY)

# Define the structure of your state
class State(TypedDict):
    topic: str
    outline: str
    essay: str

# Function to generate the essay outline
def generateOutline(state: State) -> State:
    prompt = f"""You are given a topic and you need to write an outline for the essay. 
The topic is: {state['topic']}
"""
    response = model.invoke(prompt)  # call the model
    state['outline'] = response      # assign outline to state instance
    return state

# Function to generate the essay based on the outline
def generateEssay(state: State) -> State:
    prompt = f"""You are given an outline for an essay. 
The outline is: {state['outline']}
"""
    response = model.invoke(prompt)
    state['essay'] = response
    return state

# Create the workflow graph
graph = StateGraph(State)
graph.add_node("generateOutline", generateOutline)
graph.add_node("generateEssay", generateEssay)

# Define edges between the steps
graph.add_edge(START, "generateOutline")
graph.add_edge("generateOutline", "generateEssay")
graph.add_edge("generateEssay", END)

# Compile the workflow
workflow = graph.compile()  # the variable name 'workflow' is not mandatory, but convenient

# Initial state for the workflow
initial_state: State = {"topic": "uses of AI in education", "outline": "", "essay": ""}

# Invoke the workflow
final_state = workflow.invoke(initial_state)

print("Outline:", final_state['outline'])

# Access essay from final state
print("Essay:", final_state['essay'])
