from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=1)

class InputState(TypedDict):
    question: str

class GraphState(TypedDict):
    question: str
    documents: List[str]
    messages: List[str]


# Define prompt template
prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
Your job is to answer the provided question so that even a 5 year old can understand it. 
You have provided with relevant background context to answer the question.

Question: {question} 

Context: {context}

Answer:"""
# print("Prompt Template: ", prompt)


# ------------------------------------------------------------
# Use LangGraph to create ELI5 Application
# ------------------------------------------------------------

def search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    web_docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in web_docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}

    
def explain(state: GraphState):
    """
    Generate response
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state.get("documents", [])
    formatted = prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = llm.invoke([HumanMessage(content=formatted)])
    return {"question": question, "messages": [generation]}


graph = StateGraph(GraphState, input_schema=InputState)
graph.add_node("explain", explain)
graph.add_node("search", search)
graph.add_edge(START, "search")
graph.add_edge("search", "explain")
graph.add_edge("explain", END)

eli5_working = graph.compile()

# ------------------------------------------------------------
# Creating a Buggy ELI5 Application
# ------------------------------------------------------------

buggy_prompt = """You are a professor and expert in complex technical communication.
Your job is to answer the provided question as precisely as possible, using technical language with maximal detail. 
You have provided with relevant background context to answer the question.

Question: {question} 

Context: {context}

Answer:"""

def buggy_explain(state: GraphState):
    """
    Generate response
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state.get("documents", [])
    formatted = buggy_prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = llm.invoke([HumanMessage(content=formatted)])
    return {"question": question, "messages": [generation]}

buggy_graph = StateGraph(GraphState, input_schema=InputState)
buggy_graph.add_node("explain", buggy_explain)
buggy_graph.add_node("search", search)
buggy_graph.add_edge(START, "search")
buggy_graph.add_edge("search", "explain")
buggy_graph.add_edge("explain", END)

eli5_buggy = buggy_graph.compile()

# ------------------------------------------------------------
# Creating a Flaky ELI5 Application
# ------------------------------------------------------------

flaky_prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
You must use the provided context to answer the question. If no context is available, refuse to answer the question to avoid hallucination.

Question: {question} 

Context: {context}

Answer:"""

def flaky_explain(state: GraphState):
    """
    Generate response
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state.get("documents", [])
    formatted = flaky_prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = llm.invoke([HumanMessage(content=formatted)])
    return {"question": question, "messages": [generation]}

def flaky_search(state):
    """
    Flaky search that fails to return relevant results.
    """
    question = state["question"]
    documents = state.get("documents", [])
    # Web search
    if "economics" in question:
        web_results = "No results found."
    else:
        web_docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in web_docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

flaky_graph = StateGraph(GraphState, input_schema=InputState)
flaky_graph.add_node("explain", flaky_explain)
flaky_graph.add_node("search", flaky_search)
flaky_graph.add_edge(START, "search")
flaky_graph.add_edge("search", "explain")
flaky_graph.add_edge("explain", END)

eli5_flaky = flaky_graph.compile()