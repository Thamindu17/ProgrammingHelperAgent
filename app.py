# app.py

import gradio as gr
from agent.main_agent import create_qa_chain

print("Creating the QA chain, this may take a moment...")
# 1. Create the QA chain once and store it
qa_chain = create_qa_chain()
print("✅ QA chain is ready.")


def get_answer(question):
    """
    This function takes a user's question, sends it to the agent,
    and returns the answer.
    """
    print(f"Received question: {question}")
    result = qa_chain.invoke({"query": question})
    return result["result"]

# 2. Create the Gradio Interface
iface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(lines=2, placeholder="e.g., How do I merge two DataFrames in Pandas?"),
    outputs="text",
    title="🤖 Programming Helper Agent",
    description="Ask a programming question, and the agent will answer based on its documentation knowledge.",
    allow_flagging="never"
)

# 3. Launch the Interface
if __name__ == "__main__":
    iface.launch()