# utils/dynamic_vector_store.py

import os
import pickle
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any
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
        """Initialize embedding model with graceful fallbacks.

        Priority:
        1. If env USE_REMOTE_EMBEDDINGS=true and HF key present -> use remote inference (no heavy local deps)
        2. Try local sentence-transformers model (may pull & load SciPy/sklearn stack)
        3. Fallback to a lightweight hash embedding (non-semantic but functional) so the app still works.
        """
        use_remote = os.getenv("USE_REMOTE_EMBEDDINGS", "").lower() in ("1", "true", "yes")
        force_light = os.getenv("LIGHT_EMBEDDINGS", "").lower() in ("1", "true", "yes")
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        class HashEmbedding:
            """Lightweight fallback embedding producing deterministic pseudo-vectors.
            Not semantically rich, but lets vector search function for demo/minimal mode.
            """
            def __init__(self, dim: int = 384):
                self.dim = dim

            def _hash(self, token: str) -> int:
                h = 0
                for c in token:
                    h = (h * 131 + ord(c)) & 0xFFFFFFFF
                return h

            def embed_documents(self, texts):  # noqa: D401
                return [self._embed(t) for t in texts]

            def embed_query(self, text):
                return self._embed(text)

            def _embed(self, text: str):
                import math
                vec = [0.0] * self.dim
                tokens = [t for t in text.lower().split() if t]
                if not tokens:
                    return vec
                for t in tokens:
                    idx = self._hash(t) % self.dim
                    vec[idx] += 1.0
                # L2 normalize
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                return [v / norm for v in vec]

        # 1. Remote embeddings via HuggingFace Inference
        if not force_light and use_remote and hf_api_key:
            try:
                print("🔍 Initializing remote Hugging Face Inference embeddings...")
                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
                model = os.getenv("REMOTE_EMBEDDING_MODEL", self.embedding_model)
                self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key=hf_api_key,
                    model_name=model,
                )
                print("✅ Remote inference embeddings ready")
                return
            except Exception as e:
                print(f"⚠️ Remote embedding init failed: {e}. Falling back to local.")

        # 2. Local model
        if not force_light:
            try:
                print("🔍 Initializing local embedding model (sentence-transformers)...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': 'cpu'}
                )
                print("✅ Local embedding model ready")
                return
            except KeyboardInterrupt:
                print("⏭️ Local embedding initialization interrupted by user; switching to lightweight hash embedding.")
            except Exception as e:
                print(f"⚠️ Local embedding init failed: {e}. Using lightweight hash embedding.")

        # 3. Lightweight hash fallback
        print("🔧 Using lightweight hash-based embeddings (LOW QUALITY, for minimal mode). Set LIGHT_EMBEDDINGS=0 to attempt full model.")
        self.embeddings = HashEmbedding()
    
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
        if not all_results:
            # Fallback: simple keyword scan across raw stored docs (best-effort)
            print("🩹 Vector search empty – performing keyword fallback")
            keyword_hits = self._keyword_fallback(query, store_names, k)
            if keyword_hits:
                print(f"🧪 Keyword fallback produced {len(keyword_hits)} matches")
                return keyword_hits
            else:
                print("⚠️ Keyword fallback found nothing")
        return all_results

    # ------------------------------------------------------------------
    # Fallback & Debug Helpers
    # ------------------------------------------------------------------
    def _keyword_fallback(self, query: str, store_names: List[str], k: int) -> List[Document]:
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return []
        scored: List[tuple[float, Document]] = []
        for store_name in store_names:
            vs = self.load_vector_store(store_name)
            if not vs:
                continue
            # Access underlying docs (FAISS stores in _documents / _docstore)
            try:
                # For safety handle different internal structures
                docs_iter = []
                if hasattr(vs, 'docstore') and hasattr(vs.docstore, 'search'):  # new style
                    # We don't have an index of ids here; attempt exporting
                    if hasattr(vs.docstore, 'dict'):
                        docs_iter = list(vs.docstore.dict.values())
                if not docs_iter and hasattr(vs, 'index_to_docstore_id'):
                    ids = [vs.index_to_docstore_id[i] for i in vs.index_to_docstore_id]
                    if hasattr(vs, 'docstore'):
                        docs_iter = [vs.docstore.search(i) for i in ids]
                for doc in docs_iter:
                    if not doc or not getattr(doc, 'page_content', None):
                        continue
                    text_lower = doc.page_content.lower()
                    score = sum(text_lower.count(term) for term in terms)
                    if score > 0:
                        # Clone doc with annotation
                        new_meta = dict(doc.metadata)
                        new_meta['fallback'] = 'keyword'
                        new_meta['vector_store'] = store_name
                        scored.append((score, Document(page_content=doc.page_content[:800], metadata=new_meta)))
            except Exception as e:
                print(f"⚠️ Keyword fallback error in '{store_name}': {e}")
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]

    def inspect_store_chunks(self, store_name: str, limit: int = 3) -> List[Dict]:
        """Return sample chunk metadata + preview for debugging."""
        vs = self.load_vector_store(store_name)
        if not vs:
            return []
        samples = []
        try:
            docs_iter = []
            if hasattr(vs, 'docstore') and hasattr(vs.docstore, 'dict'):
                docs_iter = list(vs.docstore.dict.values())
            if not docs_iter and hasattr(vs, 'index_to_docstore_id'):
                ids = [vs.index_to_docstore_id[i] for i in vs.index_to_docstore_id]
                if hasattr(vs, 'docstore'):
                    docs_iter = [vs.docstore.search(i) for i in ids]
            for doc in docs_iter[:limit]:
                if not doc:
                    continue
                samples.append({
                    'preview': doc.page_content[:200],
                    'metadata': doc.metadata
                })
        except Exception as e:
            print(f"⚠️ Inspect error: {e}")
        return samples
    
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