from  langgraph.graph import StateGraph, START, END
from typing  import TypedDict,Annotated
from langgraph.graph import StateGraph, START, END
from typing  import TypedDict,Annotated
from langchain_openai import ChatOpenAI
import os
from pydantic import BaseModel , Field
import operator
API_KEY = os.environ.get('OPEN_AI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)  # normal hyphens

class EssayState(TypedDict):
    content: str
    clarityOfThought: str
    depthOfAnalysis: str
    language: str


    individual_scores: Annotated[list[int], operator.add]
    finalSummary: str
    avgScore: float

class EvaluationSchema(BaseModel):
    feedback: str = Field(..., description="Detailed Feedback for the essay")
    score: int = Field(..., description="Score out of 10. Must be between 0 and 10")

# Assuming 'model' is already defined
structured_model = model.with_structured_output(schema=EvaluationSchema)


def evaluate_language(state: EssayState):
    prompt = f'Evaluate the language quality of the following essay and provide feedback and assign a score out of 10:\n{state["content"]}'
    output = structured_model.invoke(prompt)
    return {'language': output.feedback, 'individual_scores': [output.score]}


def evaluate_clarity_of_thought(state: EssayState):
    prompt = f'Evaluate the clarity of thought quality of the following essay and provide feedback and assign a score out of 10:\n{state["content"]}'
    output = structured_model.invoke(prompt)
    return {'clarityOfThought': output.feedback, 'individual_scores': [output.score]}


def evaluate_depth_of_analysis(state: EssayState):
    prompt = f'Evaluate the depth of analysis quality of the following essay and provide feedback and assign a score out of 10:\n{state["content"]}'
    output = structured_model.invoke(prompt)
    return {'depthOfAnalysis': output.feedback, 'individual_scores': [output.score]}


def evaluate_summary(state: EssayState):
    prompt = f"""Based on the following feedbacks, create a summarized feedback: {state["language"]}, {state["clarityOfThought"]}, {state["depthOfAnalysis"]}"""
    output = model.invoke(prompt)

    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])

    return {'finalSummary': output, 'avgScore': avg_score}


graph = StateGraph(EssayState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_clarity_of_thought", evaluate_clarity_of_thought)
graph.add_node("evaluate_depth_of_analysis", evaluate_depth_of_analysis)
graph.add_node("evaluate_summary", evaluate_summary)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_clarity_of_thought")
graph.add_edge(START, "evaluate_depth_of_analysis")
graph.add_edge("evaluate_depth_of_analysis", "evaluate_summary")
graph.add_edge("evaluate_summary", END)


workflow = graph.compile()
essay = """
UK Drill rap is a gritty and intense genre originating from London, known for its dark beats and raw lyrics. It reflects the struggles of urban life, often telling stories of street life and survival. The music uses fast flows and heavy bass to create a haunting atmosphere. Despite controversy, it has influenced global hip-hop culture with its unique style and energy.
"""

intial_state = {"content": essay}
finalState =workflow.invoke(intial_state)

final_summary = finalState['finalSummary']
avg_score = finalState['avgScore']
print("Final Summary:", final_summary)
print("Average Score:", avg_score)




