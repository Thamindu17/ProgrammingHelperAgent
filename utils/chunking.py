# utils/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

def load_and_chunk_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Loads a document from a file path and splits it into chunks.

    Args:
        file_path (str): The path to the text document.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of document chunks.
    """
    # 1. Load the document
    loader = TextLoader(file_path, encoding="UTF-8")
    documents = loader.load()
    
    # 2. Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # 3. Split the document into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"Successfully loaded and split {file_path} into {len(chunks)} chunks.")
    
    return chunks

# Example usage (you can run this file directly to test)
if __name__ == '__main__':
    chunks = load_and_chunk_documents('data/pandas_docs.txt')
    # Print the first chunk to see the result
    if chunks:
        print("\n--- Example Chunk ---")
        print(chunks[0].page_content)
        print("---------------------\n")