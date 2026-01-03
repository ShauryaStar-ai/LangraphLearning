
from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Literal
from langchain_openai import ChatOpenAI
import os

from pydantic import BaseModel, Field

API_KEY = os.environ.get('OPEN_AI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')


class SentimentSchema(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(description='Sentiment of the review')

class Responce(TypedDict):
    userReview: str
    sentiment:  Literal["positive", "negative"]
    diagnosis: dict
    response: str

structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

graph = StateGraph(Responce)
def findSentiment(state:Responce):
    prompt = f'For the following review find out the sentiment \n {state["userReview"]}'
    sentiment = structured_model.invoke(prompt).sentiment

    return {'sentiment': sentiment}

def checkSentiment(state:Responce)->Literal["postitve_responce", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "postitve_responce"
    else:
        return "run_diagnosis"


def postitve_responce(state: Responce):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['userReview']}\"\n
Also, kindly ask the user to leave feedback on our website."""

    response = model.invoke(prompt).content

    return {'response': response}

def run_diagnosis(state: Responce):
    prompt = f"""Diagnose the following negative review:

"{state['userReview']}"

Return issue_type, tone, and urgency.
"""
    diagnosis = structured_model2.invoke(prompt)

    return {
        "diagnosis": diagnosis.model_dump()
    }

def negetiveResponce(state:Responce):
    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message.
    """
    response = model.invoke(prompt).content

    return {'response': response}

graph.add_node("findSentiment",findSentiment)
graph.add_node("checkSentiment",checkSentiment)
graph.add_node("postitve_responce",postitve_responce)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("negetiveResponce",negetiveResponce)


graph.add_edge(START, "findSentiment")
graph.add_conditional_edges("findSentiment" , checkSentiment)
graph.add_edge("postitve_responce", END)
graph.add_edge("run_diagnosis","negetiveResponce" )
graph.add_edge("negetiveResponce", END)

workflow = graph.compile()
initalState = Responce(userReview="Your sales and product was really good")
finalState = workflow.invoke(initalState)
print(finalState["response"])


