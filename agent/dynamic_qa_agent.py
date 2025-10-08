# agent/dynamic_qa_agent.py

from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DynamicQAAgent:
    """
    Enhanced QA Agent that works with multiple documentation sources
    Gracefully handles missing API keys
    """
    
    def __init__(self):
        self.llm = None
        self.initialized = False
        self._initialize_llm()
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "sources"],
            template="""
You are a helpful programming assistant with access to multiple documentation sources.

Based on the following documentation context, please provide a comprehensive and accurate answer to the user's question.

Documentation Sources Used: {sources}

Context from documentation:
{context}

Question: {question}

Instructions:
1. Provide a clear, detailed answer based on the documentation
2. Include code examples when relevant and available in the context
3. Mention which documentation source(s) you're referencing when possible
4. If the answer spans multiple sources, organize the information clearly
5. If information is not available in the provided context, say so clearly
6. Be specific and practical in your response

Answer:
"""
        )
    
    def _initialize_llm(self):
        """Initialize LLM with graceful fallback"""
        try:
            # Try to load from config
            from config.settings import GROQ_API_KEY, LLM_MODEL
            
            if GROQ_API_KEY:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=LLM_MODEL,
                    temperature=0.1
                )
                self.initialized = True
                print("✅ LLM initialized with Groq API")
            else:
                print("⚠️ No GROQ_API_KEY found - using fallback responses")
                self.initialized = False
                
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            self.initialized = False
    
    def generate_response(self, question: str, relevant_docs: List[Document]) -> str:
        """Generate response based on relevant documents from multiple sources"""
        
        if not relevant_docs:
            return "I don't have enough information to answer that question based on the available documentation. Please try adding more documentation sources or rephrasing your question."
        
        # If LLM is not initialized, provide a structured fallback
        if not self.initialized:
            return self._generate_fallback_response(question, relevant_docs)
        
        try:
            # Determine retrieval mode (if any doc marked as fallback)
            retrieval_mode = "keyword_fallback" if any(d.metadata.get('fallback') == 'keyword' for d in relevant_docs) else "vector"
            # Organize context by source
            context_by_source = {}
            sources_used = set()
            
            for doc in relevant_docs:
                source = doc.metadata.get('vector_store', 'Unknown Source')
                source_file = doc.metadata.get('filename', doc.metadata.get('source', ''))
                sources_used.add(f"{source} ({source_file})")
                
                if source not in context_by_source:
                    context_by_source[source] = []
                context_by_source[source].append(doc.page_content)
            
            # Format context
            formatted_context = ""
            for source, contents in context_by_source.items():
                formatted_context += f"\n--- From {source} ---\n"
                formatted_context += "\n\n".join(contents[:2])  # Limit to 2 chunks per source
                formatted_context += "\n"
            
            # Create chain and generate response
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            
            response = chain.run(
                context=formatted_context,
                question=question,
                sources=", ".join(sources_used)
            )
            header = f"### Retrieval Mode: {retrieval_mode}\n\n"
            if retrieval_mode == "keyword_fallback":
                header += "_Vector similarity returned no hits; keyword fallback provided approximate matches._\n\n"
            return header + response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return self._generate_fallback_response(question, relevant_docs)
    
    def _generate_fallback_response(self, question: str, relevant_docs: List[Document]) -> str:
        """Generate a structured response when LLM is not available"""
        
        # Organize sources
        sources_info = {}
        for doc in relevant_docs:
            source = doc.metadata.get('vector_store', 'Unknown Source')
            filename = doc.metadata.get('filename', 'Unknown File')
            
            if source not in sources_info:
                sources_info[source] = {'files': set(), 'content': []}
            
            sources_info[source]['files'].add(filename)
            sources_info[source]['content'].append(doc.page_content[:300] + "...")
        
        # Build response
        response = f"## 📚 Documentation Search Results for: '{question}'\n\n"
        response += f"**Found information in {len(sources_info)} documentation source(s):**\n\n"
        
        for source, info in sources_info.items():
            response += f"### 📖 {source}\n"
            response += f"**Files:** {', '.join(info['files'])}\n\n"
            
            response += "**Relevant Content:**\n"
            for i, content in enumerate(info['content'][:2], 1):
                response += f"{i}. {content}\n\n"
            
            response += "---\n\n"
        
        response += "💡 **Note:** Add your GROQ_API_KEY to .env file for AI-powered responses that directly answer your questions!\n\n"
        response += "🔧 **Setup:** Copy .env.template to .env and add your API keys for enhanced functionality."
        
        return response
    
    def is_available(self) -> bool:
        """Check if the QA agent is fully functional"""
        return self.initialized
    
    def get_status(self) -> Dict:
        """Get status information about the QA agent"""
        from config.settings import MINIMAL_MODE, ACTIVE_FEATURES
        status = {
            'initialized': self.initialized,
            'llm_available': self.llm is not None,
            'response_mode': 'ai_powered' if self.initialized else 'documentation_search',
            'minimal_mode': MINIMAL_MODE,
            'active_features': ACTIVE_FEATURES
        }
        if not self.initialized:
            status['setup_needed'] = [
                "Add GROQ_API_KEY to .env file",
                "Get free API key from: https://console.groq.com/"
            ]
        return status

# Test the QA agent
if __name__ == "__main__":
    from langchain.schema import Document
    
    qa_agent = DynamicQAAgent()
    
    # Test documents
    test_docs = [
        Document(
            page_content="Python pandas is a data manipulation library. Use pd.merge() to join dataframes.",
            metadata={"vector_store": "pandas_docs", "filename": "merging.md"}
        ),
        Document(
            page_content="The merge function combines DataFrames based on common columns or indices.",
            metadata={"vector_store": "pandas_docs", "filename": "api_reference.md"}
        )
    ]
    
    # Test response generation
    question = "How do I merge two DataFrames in pandas?"
    response = qa_agent.generate_response(question, test_docs)
    
    print("Test Response:")
    print(response)
    print(f"\nAgent Status: {qa_agent.get_status()}")