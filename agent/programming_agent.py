"""
Main Programming Helper Agent that orchestrates the RAG pipeline
Combines document retrieval, context building, and response generation
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .retriever import VectorStore, DocumentRetriever
from .llm_interface import LLMInterface, ResponseSummarizer
from utils import FileProcessor, TextChunker, TextCleaner
from config import config


class ProgrammingHelperAgent:
    """Main agent class that orchestrates the RAG pipeline"""
    
    def __init__(self, data_dir: str = None, load_existing: bool = True):
        """
        Initialize the Programming Helper Agent
        
        Args:
            data_dir: Directory containing documentation files
            load_existing: Whether to load existing vector store
        """
        self.data_dir = Path(data_dir or config.DATA_DIR)
        
        # Initialize components
        self.vector_store = VectorStore()
        self.retriever = DocumentRetriever(self.vector_store)
        self.llm = LLMInterface()
        self.summarizer = ResponseSummarizer(self.llm)
        
        # Utility components
        self.file_processor = FileProcessor(config.SUPPORTED_EXTENSIONS)
        self.text_chunker = TextChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.text_cleaner = TextCleaner()
        
        # Agent state
        self.is_initialized = False
        self.last_query = None
        self.conversation_history = []
        
        # Try to load existing data
        if load_existing:
            self.load_knowledge_base()
        
        print("Programming Helper Agent initialized!")
        self.print_status()
    
    def initialize_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the knowledge base by processing documents and building embeddings
        
        Args:
            force_rebuild: Whether to rebuild even if existing data found
            
        Returns:
            Boolean indicating success
        """
        try:
            print("Initializing knowledge base...")
            
            # Check if data directory exists
            if not self.data_dir.exists():
                print(f"Creating data directory: {self.data_dir}")
                self.data_dir.mkdir(parents=True, exist_ok=True)
                print("Data directory created. Please add your documentation files to it.")
                return False
            
            # Check for existing vector store
            if not force_rebuild and self.vector_store.load():
                print("Loaded existing vector store")
                self.is_initialized = True
                return True
            
            # Process files in data directory
            print(f"Processing files in {self.data_dir}...")
            processed_files = self.file_processor.batch_process_directory(
                self.data_dir, 
                recursive=True
            )
            
            if not processed_files:
                print("No files found to process. Please add documentation files to the data directory.")
                return False
            
            # Create chunks from processed files
            all_chunks = []
            for file_data in processed_files:
                print(f"Chunking {file_data['file_name']}...")
                
                # Clean the text
                cleaned_content = self.text_cleaner.clean_text(
                    file_data['content'], 
                    preserve_code=True
                )
                
                # Create chunks
                chunks = self.text_chunker.chunk_text(
                    cleaned_content,
                    source_file=file_data['file_name'],
                    preserve_structure=True
                )
                
                all_chunks.extend(chunks)
            
            print(f"Created {len(all_chunks)} chunks from {len(processed_files)} files")
            
            # Add chunks to vector store
            if self.vector_store.add_chunks(all_chunks):
                # Save the vector store
                if self.vector_store.save():
                    print("Knowledge base initialized and saved successfully!")
                    self.is_initialized = True
                    return True
                else:
                    print("Failed to save vector store")
                    return False
            else:
                print("Failed to add chunks to vector store")
                return False
                
        except Exception as e:
            print(f"Error initializing knowledge base: {str(e)}")
            return False
    
    def load_knowledge_base(self) -> bool:
        """Load existing knowledge base from disk"""
        try:
            if self.vector_store.load():
                self.is_initialized = True
                print("Loaded existing knowledge base")
                return True
            else:
                print("No existing knowledge base found")
                return False
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            return False
    
    def ask_question(self, question: str, include_context: bool = True,
                    max_results: int = None) -> Dict[str, Any]:
        """
        Ask a question and get a response from the agent
        
        Args:
            question: The user's question
            include_context: Whether to include retrieved context
            max_results: Maximum number of context documents to retrieve
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_initialized:
            return {
                'success': False,
                'response': "Knowledge base not initialized. Please run initialize_knowledge_base() first.",
                'context': [],
                'query': question,
                'timestamp': datetime.now().isoformat()
            }
        
        if not self.llm.is_available():
            return {
                'success': False,
                'response': "Language model not available. Please check your configuration and API keys.",
                'context': [],
                'query': question,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            print(f"Processing question: {question}")
            
            # Retrieve relevant context
            context = []
            if include_context:
                print("Retrieving relevant documents...")
                context = self.retriever.retrieve_relevant_docs(
                    query=question,
                    top_k=max_results or config.TOP_K_RESULTS
                )
                print(f"Found {len(context)} relevant documents")
            
            # Generate response
            print("Generating response...")
            response = self.llm.generate_response(
                prompt=question,
                context=context,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            # Store in conversation history
            conversation_entry = {
                'question': question,
                'response': response,
                'context_count': len(context),
                'timestamp': datetime.now().isoformat()
            }
            self.conversation_history.append(conversation_entry)
            self.last_query = question
            
            # Prepare response data
            response_data = {
                'success': True,
                'response': response,
                'context': context,
                'query': question,
                'context_count': len(context),
                'timestamp': datetime.now().isoformat(),
                'model_used': self.llm.model_name
            }
            
            print("Response generated successfully!")
            return response_data
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            
            return {
                'success': False,
                'response': f"I encountered an error while processing your question: {error_msg}",
                'context': [],
                'query': question,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_document_summary(self, query: str = None) -> str:
        """
        Get a summary of the documents in the knowledge base
        
        Args:
            query: Optional query to focus the summary
            
        Returns:
            Summary string
        """
        if not self.is_initialized:
            return "Knowledge base not initialized."
        
        try:
            # Get sample documents
            if query:
                sample_docs = self.retriever.retrieve_relevant_docs(query, top_k=5)
            else:
                # Get some random chunks for general summary
                sample_docs = []
                if self.vector_store.chunks:
                    import random
                    sample_chunks = random.sample(
                        self.vector_store.chunks, 
                        min(5, len(self.vector_store.chunks))
                    )
                    for chunk in sample_chunks:
                        sample_docs.append({
                            'content': chunk.content,
                            'source_file': chunk.source_file,
                            'similarity_score': 1.0
                        })
            
            return self.summarizer.summarize_documents(sample_docs, query)
            
        except Exception as e:
            return f"Error creating summary: {str(e)}"
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents without generating a response
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching documents
        """
        if not self.is_initialized:
            return []
        
        try:
            return self.retriever.retrieve_relevant_docs(query, top_k)
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def add_document(self, file_path: str) -> bool:
        """
        Add a new document to the knowledge base
        
        Args:
            file_path: Path to the document to add
            
        Returns:
            Boolean indicating success
        """
        try:
            print(f"Adding document: {file_path}")
            
            # Process the file
            file_data = self.file_processor.process_file(file_path)
            if not file_data:
                print("Failed to process file")
                return False
            
            # Clean and chunk the content
            cleaned_content = self.text_cleaner.clean_text(
                file_data['content'], 
                preserve_code=True
            )
            
            chunks = self.text_chunker.chunk_text(
                cleaned_content,
                source_file=file_data['file_name'],
                preserve_structure=True
            )
            
            if not chunks:
                print("No chunks created from file")
                return False
            
            # Add to vector store
            if self.vector_store.add_chunks(chunks):
                # Save updated vector store
                if self.vector_store.save():
                    print(f"Successfully added {len(chunks)} chunks from {file_data['file_name']}")
                    return True
                else:
                    print("Failed to save updated vector store")
                    return False
            else:
                print("Failed to add chunks to vector store")
                return False
                
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent and knowledge base"""
        
        base_stats = {
            'initialized': self.is_initialized,
            'data_directory': str(self.data_dir),
            'llm_model': self.llm.model_name,
            'llm_available': self.llm.is_available(),
            'conversation_count': len(self.conversation_history),
            'last_query': self.last_query,
            'config': {
                'chunk_size': config.CHUNK_SIZE,
                'chunk_overlap': config.CHUNK_OVERLAP,
                'top_k_results': config.TOP_K_RESULTS,
                'temperature': config.TEMPERATURE
            }
        }
        
        if self.is_initialized:
            vector_stats = self.vector_store.get_stats()
            base_stats.update(vector_stats)
        
        return base_stats
    
    def print_status(self):
        """Print current status of the agent"""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print("Programming Helper Agent Status")
        print("=" * 50)
        print(f"Initialized: {stats['initialized']}")
        print(f"LLM Model: {stats['llm_model']}")
        print(f"LLM Available: {stats['llm_available']}")
        print(f"Data Directory: {stats['data_directory']}")
        
        if stats['initialized']:
            print(f"Documents Indexed: {stats.get('files_indexed', 0)}")
            print(f"Total Chunks: {stats.get('total_chunks', 0)}")
            print(f"Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        
        print(f"Conversations: {stats['conversation_count']}")
        print("=" * 50 + "\n")
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.last_query = None
        print("Conversation history reset")
    
    def export_conversation(self, file_path: str = None) -> str:
        """
        Export conversation history to JSON file
        
        Args:
            file_path: Path to save the conversation
            
        Returns:
            Path to the saved file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"conversation_export_{timestamp}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'agent_stats': self.get_stats(),
                    'conversation_history': self.conversation_history,
                    'export_timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            print(f"Conversation exported to {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error exporting conversation: {str(e)}")
            return ""