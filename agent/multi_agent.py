# agent/multi_agent.py

import operator
from typing import TypedDict, Annotated, List

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Import the tools from our previous setup
from agent.main_agent import create_qa_chain
from config.settings import GROQ_API_KEY
from langchain_groq import ChatGroq

# --- 1. Define the Agent State ---
# This is the shared "clipboard" that all agents will modify.
class AgentState(TypedDict):
    """
    Represents the state of our multi-agent workflow.
    """
    messages: Annotated[List[any], operator.add]
    context: str
    final_answer: str

# --- 2. Instantiate Agents/Tools ---
# We will reuse our existing RAG chain as the "Doc Search Agent"
doc_search_agent = create_qa_chain()

# We'll create a new, powerful LLM for the "Code Formatter Agent"
# Using a larger model here can lead to better explanations and formatting.
code_formatter_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # Latest supported model as of Sep 2025
    temperature=0
)

# --- 3. Define Agent Nodes ---
# Each node is a function that performs an action and modifies the state.

def documentation_retriever_node(state: AgentState) -> AgentState:
    """
    Agent Node: The "Librarian"
    - Takes the user's question from the state.
    - Uses the RAG chain to retrieve relevant documents.
    - Stores the raw context back into the state.
    """
    print("--- AGENT: Librarian ---")
    question = state["messages"][-1].content
    
    # We invoke our existing chain, but only care about the source documents
    result = doc_search_agent.invoke({"query": question})
    context = "\n---\n".join([doc.page_content for doc in result["source_documents"]])
    
    print("Librarian: Found relevant context.")
    state["context"] = context
    return state

def code_formatter_node(state: AgentState) -> AgentState:
    """
    Agent Node: The "Technical Writer"
    - Takes the question and the raw context.
    - Crafts a detailed prompt for the formatter LLM.
    - Generates a final, well-structured answer.
    """
    print("--- AGENT: Technical Writer ---")
    question = state["messages"][-1].content
    context = state["context"]
    
    prompt = f"""You are an expert programmer and technical writer. Your task is to provide a clear, concise, and accurate answer to the user's programming question based *only* on the provided context.

    Follow these rules:
    1.  Start with a direct explanation.
    2.  Include a runnable, well-formatted code block using markdown.
    3.  If the context mentions parameters or options (like 'how' in a merge), explain them briefly.
    4.  Do not use any information outside of the provided context.

    CONTEXT:
    ---
    {context}
    ---

    QUESTION:
    {question}

    YOUR ANSWER:
    """
    
    response = code_formatter_llm.invoke(prompt)
    print("Technical Writer: Formatted the final answer.")
    state["final_answer"] = response.content
    return state

# --- 4. Wire the Graph ---
# This is where we define the flow of control between the agents.

print("Constructing the agent workflow...")
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("retriever", documentation_retriever_node)
workflow.add_node("formatter", code_formatter_node)

# Define the edges that connect the nodes
workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "formatter")
workflow.add_edge("formatter", END) # The 'END' node signifies the workflow is complete

# Compile the graph into a runnable application
app = workflow.compile()
print("Agent workflow compiled successfully.")


# --- 5. Main Execution Loop ---
if __name__ == "__main__":
    print("\n🚀 Multi-Agent Programming Helper is ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Format the input for the graph
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Invoke the multi-agent workflow
        final_state = app.invoke(initial_state)
        
        print("\n--- FINAL ANSWER ---")
        print(final_state["final_answer"])
        print("--------------------")