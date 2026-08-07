from langgraph.graph import StateGraph
from router import route_model
from models import (
    openai_llm,
    gemini_llm_with_rag,
    groq_llm_response_with_rag,
    openai_llm_with_rag,
)

def router_node(state):
    question = state["question"]
    return {"question": question, "model": route_model(question)}

def gemini_node(state):
    try:
        answer = gemini_llm_with_rag(state["question"])
        if not answer:
            raise Exception("Empty response")
        return {"answer": answer}
    except Exception as exc:
        return {"answer": f"Gemini could not answer right now. Please try again. ({exc})"}

def groq_node(state):
    try:
        answer = groq_llm_response_with_rag(state["question"])
        if not answer:
            raise Exception("Empty response")
        return {"answer": answer}
    except Exception as exc:
        return {"answer": f"I couldn't reach Groq right now. Please check the Groq API key and try again. ({exc})"}

def openai_node(state):
    # Using RAG-augmented response
    return {"answer": openai_llm_with_rag(state["question"])}

builder = StateGraph(dict)

builder.add_node("router", router_node)
builder.add_node("gemini", gemini_node)
builder.add_node("groq", groq_node)
builder.add_node("openai", openai_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    lambda state: state["model"],
    {
        "gemini": "gemini",
        "groq": "groq"
    }
)

graph = builder.compile()
