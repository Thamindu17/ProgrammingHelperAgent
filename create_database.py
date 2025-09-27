# create_database.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.chunking import load_and_chunk_documents
from config.settings import EMBEDDING_MODEL

# Define paths
DATA_PATH = "data/pandas_docs.txt"
DB_FAISS_PATH = "embeddings/"

def create_vector_database():
    """
    Creates a FAISS vector database from the documents.
    """
    # 1. Load and chunk the documents
    print("Loading and chunking documents...")
    chunks = load_and_chunk_documents(DATA_PATH)
    
    if not chunks:
        print("No chunks were created. Please check the data source.")
        return

    # 2. Initialize the embedding model from Hugging Face
    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Use CPU for compatibility
    )

    # 3. Create the FAISS vector store from the document chunks
    # This process will take the text chunks, convert them to embeddings,
    # and store them in the FAISS index.
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)

    # 4. Save the vector store locally
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)
        
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database created and saved at: {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_vector_database()