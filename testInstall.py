from  langgraph.graph import StateGraph, START, END
from typing  import TypedDict

class State(TypedDict):
    Weight : float
    Height : float
    Catagory : str
    BMI : float
def CalculateBMI(state:State)->State:
    weight = state["Weight"]
    height = state["Height"]
    BMI = weight/(height*height)
    state["BMI"] = BMI
    return state
def CatagorizeBMI(state:State)->State:
    if state["BMI"] < 18.5:
        state["Catagory"] = "Underweight"
    elif state["BMI"] < 25:
        state["Catagory"] = "Normal"
    elif state["BMI"] < 30:
        state["Catagory"] = "Overweight"
    return state

graph = StateGraph(State)
graph.add_node("CalculateBMI", CalculateBMI)
graph.add_node("CatagorizeBMI", CatagorizeBMI)

graph.add_edge(START, "CalculateBMI")
graph.add_edge("CalculateBMI", "CatagorizeBMI")
graph.add_edge("CatagorizeBMI", END)
workflow = graph.compile()
state = workflow.invoke({"Weight": 70, "Height": 1.7})
print(state)
