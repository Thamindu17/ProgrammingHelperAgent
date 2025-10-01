# agent/main_agent.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # <-- Import ChatGroq
from langchain.chains import RetrievalQA
from config.settings import GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL

DB_FAISS_PATH = "embeddings/"


def create_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, 
        model_kwargs={'device': 'cpu'}
    )

    print("Loading vector database...")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # 3. Initialize the Groq Chat LLM
    print(f"Initializing LLM: {LLM_MODEL}")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0 # Set to 0 for factual responses
    )

    retriever = db.as_retriever(search_kwargs={'k': 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

def ask_question(qa_chain, question):
    print(f"\nQuery: '{question}'")
    result = qa_chain.invoke({"query": question})

    print("\n--- Answer ---")
    print(result["result"])
    print("--------------")

if __name__ == "__main__":
    qa = create_qa_chain()
    question = "How do I merge two DataFrames in Pandas?"
    ask_question(qa, question)