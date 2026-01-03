from langgraph.graph import StateGraph, START, END
from typing import TypedDict,Literal
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel, Field

API_KEY = os.environ.get('OPEN_AI_API_KEY')

model_gen = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)
model_evaluation = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)
model_optimization = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

class Post(TypedDict):
    topic: str
    content: str
    evaluation: Literal["approved", "needs_improvement"]
    numIterations: int
    maxIterations: int
    feedback: str


# =========================
# POST GENERATION
# =========================

def generatePost(state: Post):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day-to-day English.
""")
    ]

    response = model_gen.invoke(messages).content

    return {
        "content": response
    }


# =========================
# EVALUATION SCHEMA
# =========================

class PostEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        ..., description="The evaluation of the tweet"
    )
    feedback: str = Field(
        ..., description="The feedback on the tweet"
    )


# =========================
# POST EVALUATION
# =========================

def evaluatePost(state: Post):
    messages = [
        SystemMessage(
            content="You are a ruthless, no-laugh-given Twitter critic. "
                    "You evaluate tweets based on humor, originality, virality, and format."
        ),
        HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['content']}"

Criteria:
1. Originality
2. Humor
3. Punchiness
4. Virality potential
5. Proper tweet format (not Q&A, under 280 chars)

Auto-reject if:
- Question-answer format
- Over 280 characters
- Traditional setup–punchline joke
- Weak or deflating ending

Respond ONLY in structured format.
""")
    ]

    model_structured = model_evaluation.with_structured_output(PostEvaluation)
    response = model_structured.invoke(messages)

    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback
    }


# =========================
# OPTIMIZATION STEP
# =========================

def optimizePost(state: Post):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:

"{state['feedback']}"

Topic: "{state['topic']}"

Original Tweet:
{state['content']}

Rewrite it as a short, viral-worthy tweet.
Avoid Q&A format and stay under 280 characters.
""")
    ]

    response = model_optimization.invoke(messages).content

    return {
        "content": response,
        "numIterations": state["numIterations"] + 1
    }


# =========================
# LOOP CONTROL LOGIC
# =========================

def checkCondition(state: Post):
    if state["evaluation"] == "approved":
        return "approved"

    if state["numIterations"] >= state["maxIterations"]:
        return "approved"

    return "needs_improvement"

graph = StateGraph(Post)
graph.add_node("generatePost",generatePost)
graph.add_node("evaluatePost",evaluatePost)
graph.add_node("optimizePost",optimizePost)

graph.add_edge(START, "generatePost")
graph.add_edge("generatePost", "evaluatePost")
graph.add_conditional_edges("evaluatePost", checkCondition,{"approved":END,"needs_improvement":"optimizePost"})
graph.add_edge("optimizePost", "evaluatePost")

workflow = graph.compile()
intialState = {"topic":"AI", "maxInterations":3}
finalState = workflow.invoke(intialState)
print(finalState)


