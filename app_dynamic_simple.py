# app_dynamic_simple.py

import gradio as gr
import os
from utils.dynamic_document_processor import DynamicDocumentProcessor
from utils.dynamic_vector_store import DynamicVectorStore
from agent.dynamic_qa_agent import DynamicQAAgent
from datetime import datetime
import time
import traceback

# Configure gradio
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

class ProgrammingHelperApp:
    """Main application class for the Programming Helper Agent"""
    
    def __init__(self):
        print("🚀 Initializing Programming Helper Agent...")
        
        # Initialize components
        self.doc_processor = DynamicDocumentProcessor()
        self.vector_store = DynamicVectorStore()
        self.qa_agent = DynamicQAAgent()
        
        print("✅ Programming Helper Agent initialized!")
        self._print_status()
    
    def _print_status(self):
        """Print current status"""
        qa_status = self.qa_agent.get_status()
        vector_stats = self.vector_store.get_statistics()
        
        print(f"📊 Status:")
        print(f"   - AI Mode: {qa_status['response_mode']}")
        print(f"   - Documentation Sources: {vector_stats['total_stores']}")
        print(f"   - Total Documents: {vector_stats['total_documents']}")
        
        if not qa_status['initialized']:
            print("💡 Add GROQ_API_KEY for AI-powered responses!")
    
    def add_documentation_url(self, url: str, doc_name: str):
        """Add documentation from URL"""
        if not url.strip():
            return "❌ Please enter a URL"
        
        if not doc_name.strip():
            doc_name = url.split('/')[-1] or f"docs_{int(time.time())}"
        
        try:
            source_type = self.doc_processor.detect_source_type(url)
            documents = self.doc_processor.process_documentation(url, source_type)
            
            if documents:
                source_info = {
                    'source': url,
                    'type': source_type,
                    'method': 'url',
                    'added_at': datetime.now().isoformat()
                }
                
                success = self.vector_store.create_vector_store(documents, doc_name, source_info)
                
                if success:
                    return f"✅ Successfully added '{doc_name}' with {len(documents)} document chunks!"
                else:
                    return "❌ Failed to create vector store"
            else:
                return f"❌ No content extracted from {url}"
                
        except Exception as e:
            print(f"Error adding documentation: {e}")
            return f"❌ Error: {str(e)}"
    
    def chat_with_docs(self, message: str, history: list):
        """Chat with documentation sources"""
        if not message.strip():
            return history, ""
        
        try:
            available_stores = list(self.vector_store.get_available_stores().keys())
            
            if not available_stores:
                response = "❌ No documentation available. Please add some documentation first!"
            else:
                # Search documentation
                relevant_docs = self.vector_store.search_multiple_stores(
                    message, 
                    store_names=available_stores, 
                    k=4
                )
                
                if relevant_docs:
                    # Generate response
                    response = self.qa_agent.generate_response(message, relevant_docs)
                else:
                    response = "❌ No relevant information found in the documentation."
            
            # Add to history
            history.append([message, response])
            
        except Exception as e:
            print(f"Error in chat: {e}")
            response = f"❌ Error: {str(e)}"
            history.append([message, response])
        
        return history, ""
    
    def get_available_docs(self):
        """Get list of available documentation"""
        available_stores = self.vector_store.get_available_stores()
        
        if not available_stores:
            return "No documentation sources available."
        
        result = "📚 Available Documentation Sources:\n\n"
        for store_name, info in available_stores.items():
            source = info['source_info'].get('source', 'Unknown')
            doc_count = info['document_count']
            result += f"📖 {store_name}\n"
            result += f"   Source: {source}\n"
            result += f"   Documents: {doc_count}\n\n"
        
        return result
    
    def get_setup_info(self):
        """Get setup information"""
        qa_status = self.qa_agent.get_status()
        
        info = "🔧 Setup Information\n\n"
        
        if qa_status['initialized']:
            info += "✅ AI Mode: Fully functional with GROQ API\n\n"
        else:
            info += "⚠️ Fallback Mode: Documentation search only\n"
            info += "💡 Add GROQ_API_KEY to .env for AI responses\n\n"
        
        info += "🔑 Free API Keys Setup:\n\n"
        info += "1. GROQ API (Essential):\n"
        info += "   🔗 https://console.groq.com/\n"
        info += "   📝 Add: GROQ_API_KEY=your_key\n\n"
        
        info += "2. Judge0 (Code Execution):\n"
        info += "   🔗 https://rapidapi.com/judge0-official/api/judge0-ce\n"
        info += "   📝 Add: JUDGE0_API_KEY=your_rapidapi_key\n\n"
        
        info += "3. GitHub (Code Search):\n"
        info += "   🔗 https://github.com/settings/tokens\n"
        info += "   📝 Add: GITHUB_TOKEN=your_token\n\n"
        
        info += "📝 Quick Setup:\n"
        info += "1. Copy .env.template to .env\n"
        info += "2. Add your API keys\n"
        info += "3. Restart the app\n"
        info += "4. Run: python setup_checker.py\n"
        
        return info

def main():
    """Main function to run the application"""
    print("🚀 Starting Programming Helper Agent...")
    
    # Create app
    app = ProgrammingHelperApp()
    
    # Create simple interface
    with gr.Blocks(title="🤖 Programming Helper Agent") as interface:
        
        gr.Markdown("# 🤖 Programming Helper Agent - Dynamic Documentation")
        gr.Markdown("*Add your own documentation and get AI-powered answers!*")
        
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Ask about your documentation...")
            clear = gr.Button("Clear")
            
            msg.submit(app.chat_with_docs, [msg, chatbot], [chatbot, msg])
            clear.click(lambda: [], outputs=chatbot)
        
        with gr.Tab("📤 Add Documentation"):
            gr.Markdown("### Add Documentation from URL")
            
            url_input = gr.Textbox(
                label="Documentation URL",
                placeholder="https://docs.python.org/3/library/pandas.html"
            )
            name_input = gr.Textbox(
                label="Name",
                placeholder="pandas_docs"
            )
            add_btn = gr.Button("🚀 Add Documentation")
            result_output = gr.Textbox(label="Result")
            
            add_btn.click(
                app.add_documentation_url,
                inputs=[url_input, name_input],
                outputs=result_output
            )
            
            gr.Markdown("""
            **Supported Sources:**
            - 📄 Documentation websites
            - 🐙 GitHub repositories  
            - 📁 GitHub files
            - 📝 Text/Markdown files
            
            **Examples:**
            - `https://docs.python.org/3/library/`
            - `https://github.com/pandas-dev/pandas`
            - `https://raw.githubusercontent.com/user/repo/main/README.md`
            """)
        
        with gr.Tab("📚 Documentation"):
            gr.Markdown("### Available Documentation Sources")
            docs_display = gr.Textbox(
                label="Documentation Sources",
                lines=10,
                value=app.get_available_docs()
            )
            refresh_btn = gr.Button("🔄 Refresh")
            refresh_btn.click(app.get_available_docs, outputs=docs_display)
        
        with gr.Tab("🔧 Setup"):
            gr.Markdown("### Setup Instructions")
            setup_display = gr.Textbox(
                label="Setup Information",
                lines=15,
                value=app.get_setup_info()
            )
            check_btn = gr.Button("🔍 Check API Keys")
            
            def check_api_keys():
                try:
                    from setup_checker import APIKeyChecker
                    checker = APIKeyChecker()
                    results = checker.check_all_keys()
                    
                    output = "🔑 API Key Status:\n\n"
                    for service, info in results.items():
                        output += f"{info['message']}\n"
                    
                    return output
                except Exception as e:
                    return f"Error checking API keys: {str(e)}"
            
            check_btn.click(check_api_keys, outputs=setup_display)
    
    # Launch
    interface.launch(
        server_name="127.0.0.1",
        server_port=7863,
        share=False
    )

if __name__ == "__main__":
    main()