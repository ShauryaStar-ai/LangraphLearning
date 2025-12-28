from  langgraph.graph import StateGraph, START, END
from typing  import TypedDict,Annotated
from langgraph.graph import StateGraph, START, END
from typing  import TypedDict,Annotated,Dict, Tuple, Optional
import math
from langchain_openai import ChatOpenAI
import os
from pydantic import BaseModel , Field
import operator
API_KEY = os.environ.get('OPEN_AI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

class EquationState(TypedDict):
    a : int
    b : int
    c : int

    standardForm : str
    dicriminant : float
    solutions: Optional[Tuple[float, ...]]  # colon, not equals


def equation_to_string(state: EquationState):

    return {"standardForm": f"{state['a']}x^2 + {state['b']}x + {state['c']} = 0"}

def findDiscriminant(state: EquationState):
    return { "dicriminant" :state["b"] ** 2 - 4 * state["a"] * state["c"]}


def find_two_real_roots(state: "EquationState") -> Dict:

    disc = state["dicriminant"]

    sqrt_d = math.sqrt(disc)
    denom = 2 * state["a"]
    root1 = (-state["b"] + sqrt_d) / denom
    root2 = (-state["b"] - sqrt_d) / denom
    return {"solutions": (root1, root2)}

def find_single_real_root(state: "EquationState") -> Dict:
    if state["a"] == 0:
        return {"solutions": None, "error": "Not a quadratic equation (a == 0)"}
    disc = state["dicriminant"]
    if disc != 0:
        return {"solutions": None, "error": "Discriminant is not zero (no single repeated root)"}
    root = -state["b"] / (2 * state["a"])
    return {"solutions": (root,)}

def find_no_real_roots(state: "EquationState") -> Dict:

    return {"solutions": None, "error": "No real roots (discriminant < 0)"}

def checkWhichSolution(state: "EquationState"):
    if state["dicriminant"] > 0:
        return "find_two_real_roots"
    elif state["dicriminant"] == 0:
        return "find_single_real_root"
    else:
        return "find_no_real_roots"

graph = StateGraph(EquationState)
graph.add_node("equation_to_string",equation_to_string)
graph.add_node("findDiscriminant",findDiscriminant)
graph.add_node("find_two_real_roots",find_two_real_roots)
graph.add_node("find_single_real_root",find_single_real_root)
graph.add_node("find_no_real_roots",find_no_real_roots)
graph.add_node("checkWhichSolution",checkWhichSolution)


graph.add_edge(START, "equation_to_string")
graph.add_edge("equation_to_string", "findDiscriminant")
graph.add_conditional_edges("findDiscriminant",checkWhichSolution)

graph.add_edge("checkWhichSolution","find_two_real_roots")
graph.add_edge("checkWhichSolution","find_single_real_root")
graph.add_edge("checkWhichSolution","find_no_real_roots")
graph.add_edge("find_two_real_roots",END)
workflow = graph.compile()

intialState = {"a": 1, "b": 2, "c": 0}

finalState = workflow.invoke(intialState)
print(finalState)

