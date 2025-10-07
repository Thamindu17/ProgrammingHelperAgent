# utils/dynamic_vector_store.py

import os
import pickle
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from datetime import datetime

class DynamicVectorStore:
    """
    Manage multiple vector stores for different documentation sources
    Works without API keys using local embeddings
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_stores = {}
        self.metadata_file = "data/vector_stores_metadata.json"
        self.vector_stores_dir = "data/vector_stores"
        
        # Create directories if they don't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs(self.vector_stores_dir, exist_ok=True)
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Load existing metadata
        self.load_metadata()
    
    def _initialize_embeddings(self):
        """Initialize embedding model (works offline)"""
        try:
            print("🔍 Initializing embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            print("✅ Embedding model ready")
        except Exception as e:
            print(f"❌ Error initializing embeddings: {e}")
            raise e
    
    def load_metadata(self):
        """Load metadata about existing vector stores"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
            print(f"📊 Loaded metadata for {len(self.metadata)} vector stores")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            self.metadata = {}
    
    def save_metadata(self):
        """Save metadata about vector stores"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            print("💾 Metadata saved")
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
    
    def create_vector_store(self, documents: List[Document], store_name: str, source_info: Dict) -> bool:
        """Create a new vector store from documents"""
        try:
            if not documents:
                print("⚠️ No documents to process")
                return False
            
            print(f"🏗️ Creating vector store '{store_name}' with {len(documents)} documents...")
            
            # Create FAISS vector store
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store
            store_path = os.path.join(self.vector_stores_dir, store_name)
            vector_store.save_local(store_path)
            
            # Update metadata
            self.metadata[store_name] = {
                'source_info': source_info,
                'document_count': len(documents),
                'created_at': datetime.now().isoformat(),
                'path': store_path,
                'embedding_model': self.embedding_model
            }
            
            # Cache in memory
            self.vector_stores[store_name] = vector_store
            
            # Save metadata
            self.save_metadata()
            
            print(f"✅ Successfully created vector store '{store_name}' with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return False
    
    def load_vector_store(self, store_name: str) -> Optional[FAISS]:
        """Load a vector store by name"""
        try:
            if store_name in self.vector_stores:
                return self.vector_stores[store_name]
            
            if store_name in self.metadata:
                store_path = self.metadata[store_name]['path']
                if os.path.exists(store_path):
                    vector_store = FAISS.load_local(
                        store_path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.vector_stores[store_name] = vector_store
                    print(f"📚 Loaded vector store '{store_name}'")
                    return vector_store
                else:
                    print(f"❌ Vector store path not found: {store_path}")
            
            return None
        except Exception as e:
            print(f"❌ Error loading vector store '{store_name}': {e}")
            return None
    
    def get_available_stores(self) -> Dict:
        """Get list of available vector stores"""
        return self.metadata
    
    def delete_vector_store(self, store_name: str) -> bool:
        """Delete a vector store"""
        try:
            if store_name in self.metadata:
                # Remove from disk
                store_path = self.metadata[store_name]['path']
                if os.path.exists(store_path):
                    import shutil
                    shutil.rmtree(store_path)
                
                # Remove from metadata
                del self.metadata[store_name]
                
                # Remove from memory cache
                if store_name in self.vector_stores:
                    del self.vector_stores[store_name]
                
                self.save_metadata()
                print(f"🗑️ Deleted vector store '{store_name}'")
                return True
            else:
                print(f"⚠️ Vector store '{store_name}' not found")
                return False
        except Exception as e:
            print(f"❌ Error deleting vector store: {e}")
            return False
    
    def search_multiple_stores(self, query: str, store_names: List[str] = None, k: int = 4) -> List[Document]:
        """Search across multiple vector stores"""
        if store_names is None:
            store_names = list(self.metadata.keys())
        
        all_results = []
        
        for store_name in store_names:
            vector_store = self.load_vector_store(store_name)
            if vector_store:
                try:
                    results = vector_store.similarity_search(query, k=k)
                    # Add store name to metadata
                    for result in results:
                        result.metadata['vector_store'] = store_name
                    all_results.extend(results)
                    print(f"🔍 Found {len(results)} results in '{store_name}'")
                except Exception as e:
                    print(f"❌ Error searching store '{store_name}': {e}")
        
        print(f"🎯 Total search results: {len(all_results)}")
        return all_results
    
    def get_store_info(self, store_name: str) -> Optional[Dict]:
        """Get detailed information about a specific store"""
        if store_name in self.metadata:
            info = self.metadata[store_name].copy()
            
            # Add current status
            vector_store = self.load_vector_store(store_name)
            info['status'] = 'loaded' if vector_store else 'error'
            info['in_memory'] = store_name in self.vector_stores
            
            return info
        return None
    
    def update_store_metadata(self, store_name: str, additional_info: Dict):
        """Update metadata for a store"""
        if store_name in self.metadata:
            self.metadata[store_name].update(additional_info)
            self.save_metadata()
    
    def get_statistics(self) -> Dict:
        """Get statistics about all vector stores"""
        total_docs = sum(info['document_count'] for info in self.metadata.values())
        
        stats = {
            'total_stores': len(self.metadata),
            'total_documents': total_docs,
            'stores_in_memory': len(self.vector_stores),
            'embedding_model': self.embedding_model,
            'storage_path': self.vector_stores_dir
        }
        
        # Group by source type
        source_types = {}
        for store_info in self.metadata.values():
            source_type = store_info['source_info'].get('type', 'unknown')
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        stats['source_types'] = source_types
        
        return stats

# Test the vector store
if __name__ == "__main__":
    from langchain.schema import Document
    
    # Create test vector store
    vector_store = DynamicVectorStore()
    
    # Test documents
    test_docs = [
        Document(
            page_content="This is a test document about Python programming.",
            metadata={"source": "test", "type": "test"}
        ),
        Document(
            page_content="Python is a powerful programming language used for web development.",
            metadata={"source": "test", "type": "test"}
        )
    ]
    
    # Test creating a vector store
    success = vector_store.create_vector_store(
        test_docs, 
        "test_store", 
        {"source": "test", "type": "test"}
    )
    
    if success:
        # Test searching
        results = vector_store.search_multiple_stores("Python programming", ["test_store"])
        print(f"Search test: Found {len(results)} results")
        
        # Test statistics
        stats = vector_store.get_statistics()
        print(f"Statistics: {stats}")
    
    print("Vector store test completed")