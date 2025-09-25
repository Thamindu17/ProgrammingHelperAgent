"""
Vector store and retrieval system for the Programming Helper Agent
Handles document embedding, indexing, and similarity search using FAISS
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import asdict

# Import for embeddings (will need to be installed)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from utils.text_chunking import TextChunk
from config import config


class VectorStore:
    """Vector store for document embeddings using FAISS"""
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize the vector store
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.embedding_dim = None
        
        # Load model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embedding model: {self.embedding_model_name}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                print(f"Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                print(f"Error loading embedding model: {str(e)}")
        else:
            print("SentenceTransformers not available. Install with: pip install sentence-transformers")
    
    def add_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Add text chunks to the vector store
        
        Args:
            chunks: List of TextChunk objects to add
            
        Returns:
            Boolean indicating success
        """
        if not self.embedding_model:
            print("Embedding model not available")
            return False
        
        if not chunks:
            print("No chunks provided")
            return False
        
        try:
            print(f"Embedding {len(chunks)} chunks...")
            
            # Extract text content for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Initialize FAISS index if needed
            if self.index is None:
                if not FAISS_AVAILABLE:
                    print("FAISS not available. Install with: pip install faiss-cpu")
                    return False
                
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))
            
            # Store chunks and metadata
            self.chunks.extend(chunks)
            for i, chunk in enumerate(chunks):
                metadata = {
                    'chunk_index': len(self.chunk_metadata),
                    'original_chunk': asdict(chunk),
                    'embedding_model': self.embedding_model_name,
                    'added_at': str(pd.Timestamp.now()) if 'pd' in globals() else str(len(self.chunk_metadata))
                }
                self.chunk_metadata.append(metadata)
            
            print(f"Successfully added {len(chunks)} chunks to vector store")
            print(f"Total chunks in store: {len(self.chunks)}")
            
            return True
            
        except Exception as e:
            print(f"Error adding chunks to vector store: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5, 
               similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query: Search query text
            top_k: Number of similar chunks to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        if not self.embedding_model or not self.index:
            print("Vector store not initialized")
            return []
        
        try:
            # Embed the query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            similarities, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(top_k, len(self.chunks))
            )
            
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if similarity >= similarity_threshold:
                    chunk = self.chunks[idx]
                    result = {
                        'chunk': chunk,
                        'similarity_score': float(similarity),
                        'chunk_id': chunk.chunk_id,
                        'source_file': chunk.source_file,
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'is_code': chunk.is_code,
                        'language': chunk.language
                    }
                    results.append(result)
            
            print(f"Found {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            return []
    
    def save(self, index_path: str = None, metadata_path: str = None) -> bool:
        """
        Save the vector store to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
            
        Returns:
            Boolean indicating success
        """
        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.METADATA_PATH
        
        try:
            # Create directories if needed
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, index_path)
                print(f"Saved FAISS index to {index_path}")
            
            # Save metadata and chunks
            save_data = {
                'chunks': [asdict(chunk) for chunk in self.chunks],
                'chunk_metadata': self.chunk_metadata,
                'embedding_model': self.embedding_model_name,
                'embedding_dim': self.embedding_dim,
                'total_chunks': len(self.chunks)
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved metadata to {metadata_path}")
            return True
            
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            return False
    
    def load(self, index_path: str = None, metadata_path: str = None) -> bool:
        """
        Load the vector store from disk
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
            
        Returns:
            Boolean indicating success
        """
        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.METADATA_PATH
        
        try:
            # Load FAISS index
            if os.path.exists(index_path) and FAISS_AVAILABLE:
                self.index = faiss.read_index(index_path)
                print(f"Loaded FAISS index from {index_path}")
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                # Reconstruct chunks from saved data
                self.chunks = []
                for chunk_data in save_data.get('chunks', []):
                    chunk = TextChunk(**chunk_data)
                    self.chunks.append(chunk)
                
                self.chunk_metadata = save_data.get('chunk_metadata', [])
                self.embedding_dim = save_data.get('embedding_dim', self.embedding_dim)
                
                print(f"Loaded {len(self.chunks)} chunks from {metadata_path}")
                return True
            else:
                print(f"Metadata file {metadata_path} not found")
                return False
                
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.chunks:
            return {'total_chunks': 0, 'status': 'empty'}
        
        # Count by source file
        file_counts = {}
        language_counts = {}
        
        for chunk in self.chunks:
            # File counts
            file_counts[chunk.source_file] = file_counts.get(chunk.source_file, 0) + 1
            
            # Language counts
            if chunk.language:
                language_counts[chunk.language] = language_counts.get(chunk.language, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dim,
            'files_indexed': len(file_counts),
            'file_counts': file_counts,
            'language_distribution': language_counts,
            'has_faiss_index': self.index is not None,
            'index_size': self.index.ntotal if self.index else 0
        }


class DocumentRetriever:
    """High-level interface for document retrieval"""
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize the document retriever
        
        Args:
            vector_store: VectorStore instance to use
        """
        self.vector_store = vector_store or VectorStore()
    
    def retrieve_relevant_docs(self, query: str, top_k: int = None, 
                              filter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_params: Additional filtering parameters
            
        Returns:
            List of relevant document chunks with metadata
        """
        top_k = top_k or config.TOP_K_RESULTS
        similarity_threshold = config.SIMILARITY_THRESHOLD
        
        # Get initial results from vector search
        results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # Get more results for filtering
            similarity_threshold=similarity_threshold
        )
        
        # Apply additional filtering if specified
        if filter_params:
            results = self._apply_filters(results, filter_params)
        
        # Limit to requested number
        results = results[:top_k]
        
        # Enhance results with context
        enhanced_results = []
        for result in results:
            enhanced_result = {
                **result,
                'relevance_explanation': self._explain_relevance(query, result),
                'context_window': self._get_context_window(result['chunk'])
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply additional filters to search results"""
        filtered_results = []
        
        for result in results:
            include = True
            
            # Filter by file type
            if 'file_extension' in filters:
                if result['chunk'].source_file.endswith(filters['file_extension']):
                    pass
                else:
                    include = False
            
            # Filter by programming language
            if 'language' in filters:
                if result['chunk'].language == filters['language']:
                    pass
                else:
                    include = False
            
            # Filter by content type (code vs text)
            if 'is_code' in filters:
                if result['chunk'].is_code == filters['is_code']:
                    pass
                else:
                    include = False
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _explain_relevance(self, query: str, result: Dict[str, Any]) -> str:
        """Generate a brief explanation of why this result is relevant"""
        # This is a simple implementation - could be enhanced with more sophisticated analysis
        similarity = result['similarity_score']
        chunk = result['chunk']
        
        explanation_parts = []
        
        if similarity > 0.8:
            explanation_parts.append("High similarity match")
        elif similarity > 0.6:
            explanation_parts.append("Good similarity match")
        else:
            explanation_parts.append("Moderate similarity match")
        
        if chunk.is_code:
            explanation_parts.append(f"Contains {chunk.language or 'code'} examples")
        
        if chunk.metadata.get('structure_type') == 'header':
            explanation_parts.append("From section header")
        
        return "; ".join(explanation_parts)
    
    def _get_context_window(self, chunk: TextChunk, window_size: int = 200) -> str:
        """Get a context window around the chunk for better understanding"""
        # This is a simplified implementation
        # In a full implementation, you might want to get surrounding chunks
        content = chunk.content
        
        if len(content) <= window_size * 2:
            return content
        
        # Get first and last parts of the chunk
        start = content[:window_size]
        end = content[-window_size:]
        
        return f"{start}...\n\n...{end}"