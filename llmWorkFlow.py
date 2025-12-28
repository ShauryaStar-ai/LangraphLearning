from  langgraph.graph import StateGraph, START, END
from typing  import TypedDict
from langchain_openai import ChatOpenAI
import os

API_KEY = os.environ.get('OPEN_AI_API_KEY')
model = ChatOpenAI(api_key=API_KEY)
class LLMState(TypedDict):
    question : str
    answer : str

def llm_qa(state: LLMState) -> LLMState:

    # extract the question from state
    question = state['question']

    # form a prompt
    prompt = f'Answer the following question {question}'

    # ask that question to the LLM
    answer = model.invoke(prompt).content

    # update the answer in the state
    state['answer'] = answer

    return state

graph = StateGraph(LLMState)
graph.add_node("LLMQA", llm_qa)

graph.add_edge(START, "LLMQA")
graph.add_edge("LLMQA", END)

workflow = graph.compile()


initial_state = {"question": "What is the capital of France?"}

final_state = workflow.invoke(initial_state)
print(final_state['answer'])



