"""
Configuration module for Programming Helper Agent
Handles API keys, model settings, and other configuration parameters
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Programming Helper Agent"""
    
    # API Keys - Load from environment variables for security
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')  # Can be changed to 'gpt-4', local models, etc.
    
    # Vector Database Settings
    VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'faiss')  # Options: 'faiss', 'chroma', 'pinecone'
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', './embeddings/faiss_index.bin')
    EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH', './embeddings/embeddings.pkl')
    METADATA_PATH = os.getenv('METADATA_PATH', './embeddings/metadata.json')
    
    # Text Processing Settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))  # Characters per chunk
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))  # Overlap between chunks
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4000'))  # Max tokens for LLM response
    
    # Data Directory Settings
    DATA_DIR = os.getenv('DATA_DIR', './data')
    SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.md', '.py', '.js', '.html', '.css']
    
    # Retrieval Settings
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))  # Number of similar chunks to retrieve
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))  # Minimum similarity score
    
    # Agent Behavior Settings
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))  # LLM creativity (0.0 to 1.0)
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))  # Max API call retries
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration settings and return status
        Returns dict with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check API keys
        if not cls.OPENAI_API_KEY and cls.LLM_MODEL.startswith('gpt'):
            validation_results['errors'].append('OpenAI API key required for GPT models')
            validation_results['valid'] = False
        
        # Check paths
        if not os.path.exists(cls.DATA_DIR):
            validation_results['warnings'].append(f'Data directory {cls.DATA_DIR} does not exist')
        
        # Check parameter ranges
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 1:
            validation_results['errors'].append('Temperature must be between 0.0 and 1.0')
            validation_results['valid'] = False
        
        if cls.CHUNK_SIZE <= cls.CHUNK_OVERLAP:
            validation_results['errors'].append('Chunk size must be larger than chunk overlap')
            validation_results['valid'] = False
        
        return validation_results
    
    @classmethod
    def print_config(cls):
        """Print current configuration (excluding sensitive data)"""
        print("=== Programming Helper Agent Configuration ===")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Vector DB Type: {cls.VECTOR_DB_TYPE}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Top-K Results: {cls.TOP_K_RESULTS}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print("=" * 45)

# Default configuration instance
config = Config()