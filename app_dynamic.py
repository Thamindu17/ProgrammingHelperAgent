# app_dynamic.py

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
        
        # App state
        self.chat_history = []
        
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
    
    def add_documentation_url(self, url: str, doc_name: str, progress=gr.Progress()):
        """Add documentation from URL"""
        if not url.strip():
            return "❌ Please enter a URL", self._get_available_docs_html()
        
        if not doc_name.strip():
            doc_name = url.split('/')[-1] or f"docs_{int(time.time())}"
        
        try:
            progress(0.1, desc="Detecting source type...")
            source_type = self.doc_processor.detect_source_type(url)
            
            progress(0.3, desc=f"Processing {source_type}...")
            documents = self.doc_processor.process_documentation(url, source_type)
            
            if documents:
                progress(0.7, desc="Creating vector store...")
                source_info = {
                    'source': url,
                    'type': source_type,
                    'method': 'url',
                    'added_at': datetime.now().isoformat()
                }
                
                success = self.vector_store.create_vector_store(documents, doc_name, source_info)
                
                if success:
                    progress(1.0, desc="Complete!")
                    return f"✅ Successfully added '{doc_name}' with {len(documents)} document chunks!", self._get_available_docs_html()
                else:
                    return "❌ Failed to create vector store", self._get_available_docs_html()
            else:
                return f"❌ No content extracted from {url}", self._get_available_docs_html()
                
        except Exception as e:
            print(f"Error adding documentation: {e}")
            traceback.print_exc()
            return f"❌ Error: {str(e)}", self._get_available_docs_html()
    
    def add_documentation_file(self, file, doc_name: str, progress=gr.Progress()):
        """Add documentation from uploaded file"""
        if file is None:
            return "❌ Please upload a file", self._get_available_docs_html()
        
        if not doc_name.strip():
            doc_name = file.name.split('.')[0]
        
        try:
            progress(0.2, desc="Reading file...")
            
            # Read file content
            if file.name.endswith('.pdf'):
                return "❌ PDF support requires additional libraries. Please use text/markdown files.", self._get_available_docs_html()
            
            file_content = file.read()
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            progress(0.5, desc="Processing content...")
            documents = self.doc_processor.process_uploaded_file(file, file_content)
            
            if documents:
                progress(0.8, desc="Creating vector store...")
                source_info = {
                    'source': file.name,
                    'type': 'uploaded_file',
                    'method': 'upload',
                    'added_at': datetime.now().isoformat()
                }
                
                success = self.vector_store.create_vector_store(documents, doc_name, source_info)
                
                if success:
                    progress(1.0, desc="Complete!")
                    return f"✅ Successfully uploaded '{doc_name}' with {len(documents)} document chunks!", self._get_available_docs_html()
                else:
                    return "❌ Failed to create vector store", self._get_available_docs_html()
            else:
                return "❌ No content extracted from file", self._get_available_docs_html()
                
        except Exception as e:
            print(f"Error uploading file: {e}")
            return f"❌ Error: {str(e)}", self._get_available_docs_html()
    
    def delete_documentation(self, doc_name: str):
        """Delete a documentation source"""
        if not doc_name:
            return "❌ Please select a documentation source to delete", self._get_available_docs_html()
        
        try:
            success = self.vector_store.delete_vector_store(doc_name)
            if success:
                return f"✅ Deleted '{doc_name}'", self._get_available_docs_html()
            else:
                return f"❌ Failed to delete '{doc_name}'", self._get_available_docs_html()
        except Exception as e:
            return f"❌ Error deleting: {str(e)}", self._get_available_docs_html()
    
    def chat_with_docs(self, message: str, selected_docs: list, history: list):
        """Chat with selected documentation sources"""
        if not message.strip():
            return history, ""
        
        # Add user message to history
        history.append([message, None])
        
        try:
            available_stores = list(self.vector_store.get_available_stores().keys())
            
            if not available_stores:
                response = "❌ No documentation available. Please add some documentation first!"
            elif not selected_docs:
                response = "❌ Please select at least one documentation source to search."
            else:
                # Search documentation
                relevant_docs = self.vector_store.search_multiple_stores(
                    message, 
                    store_names=selected_docs, 
                    k=4
                )
                
                if relevant_docs:
                    # Generate response
                    response = self.qa_agent.generate_response(message, relevant_docs)
                    
                    # Add source information if in fallback mode
                    if not self.qa_agent.is_available():
                        source_info = self._format_sources(relevant_docs)
                        response += f"\n\n📚 **Sources searched:** {source_info}"
                else:
                    response = "❌ No relevant information found in the selected documentation sources."
            
            # Update history
            history[-1][1] = response
            
        except Exception as e:
            print(f"Error in chat: {e}")
            traceback.print_exc()
            response = f"❌ Error: {str(e)}"
            history[-1][1] = response
        
        return history, ""
    
    def _format_sources(self, docs):
        """Format source information"""
        sources = set()
        for doc in docs:
            store = doc.metadata.get('vector_store', 'Unknown')
            filename = doc.metadata.get('filename', '')
            if filename:
                sources.add(f"{store} ({filename})")
            else:
                sources.add(store)
        return ", ".join(sources)
    
    def _get_available_docs_html(self):
        """Get HTML formatted list of available documentation"""
        available_stores = self.vector_store.get_available_stores()
        
        if not available_stores:
            return "<p><em>No documentation sources available.</em></p>"
        
        html = "<div style='max-height: 300px; overflow-y: auto;'>"
        html += "<h4>📚 Available Documentation Sources:</h4>"
        
        for store_name, info in available_stores.items():
            source = info['source_info'].get('source', 'Unknown')
            doc_count = info['document_count']
            created = info['created_at'][:16].replace('T', ' ')
            doc_type = info['source_info'].get('type', 'unknown')
            
            html += f"""
            <div style='border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                <strong>📖 {store_name}</strong><br>
                <small>
                    📄 {doc_count} documents | 
                    🔗 <a href='{source}' target='_blank'>{source[:50]}...</a><br>
                    📅 Added: {created} | 
                    🏷️ Type: {doc_type}
                </small>
            </div>
            """
        
        html += "</div>"
        return html
    
    def get_setup_instructions(self):
        """Get setup instructions for API keys"""
        qa_status = self.qa_agent.get_status()
        
        html = """
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
        <h3>🚀 Setup Instructions</h3>
        
        <h4>🔧 Current Status:</h4>
        """
        
        if qa_status['initialized']:
            html += "<p>✅ <strong>AI Mode:</strong> Fully functional with GROQ API</p>"
        else:
            html += "<p>⚠️ <strong>Fallback Mode:</strong> Documentation search only</p>"
        
        html += """
        <h4>🔑 API Keys Setup (All FREE!):</h4>
        
        <div style='margin: 10px 0;'>
        <strong>1. Essential - GROQ API (AI Responses)</strong><br>
        🔗 <a href='https://console.groq.com/' target='_blank'>Get Free API Key</a><br>
        📝 Add to .env: <code>GROQ_API_KEY=your_key_here</code>
        </div>
        
        <div style='margin: 10px 0;'>
        <strong>2. Recommended - Judge0 (Code Execution)</strong><br>
        🔗 <a href='https://rapidapi.com/judge0-official/api/judge0-ce' target='_blank'>Get Free API Key</a><br>
        📝 Add to .env: <code>JUDGE0_API_KEY=your_rapidapi_key</code><br>
        💰 Free: 50 code executions per day
        </div>
        
        <div style='margin: 10px 0;'>
        <strong>3. High Value - GitHub Token</strong><br>
        🔗 <a href='https://github.com/settings/tokens' target='_blank'>Create Personal Token</a><br>
        📝 Add to .env: <code>GITHUB_TOKEN=your_token</code><br>
        💰 Free: 5,000 API requests per hour
        </div>
        
        <div style='margin: 10px 0;'>
        <strong>4. Optional - Hugging Face</strong><br>
        🔗 <a href='https://huggingface.co/settings/tokens' target='_blank'>Get Token</a><br>
        📝 Add to .env: <code>HUGGINGFACEHUB_API_TOKEN=your_token</code>
        </div>
        
        <h4>📝 Quick Setup:</h4>
        <ol>
        <li>Copy <code>.env.template</code> to <code>.env</code></li>
        <li>Add your API keys to the <code>.env</code> file</li>
        <li>Restart the application</li>
        <li>Run <code>python setup_checker.py</code> to verify</li>
        </ol>
        
        <p><strong>💡 Tip:</strong> The app works without API keys but gets much better with them!</p>
        </div>
        """
        
        return html
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="🤖 Programming Helper Agent",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {max-width: 1200px; margin: 0 auto;}
            .header {text-align: center; margin-bottom: 20px;}
            """
        ) as interface:
            
            # Header
            gr.Markdown(
                """
                # 🤖 Programming Helper Agent - Dynamic Documentation
                
                *Add your own documentation sources and get AI-powered answers!*
                
                **✨ Features:** Dynamic documentation loading • Multi-source search • AI-powered responses • Code execution (with API keys)
                """,
                elem_classes=["header"]
            )
            
            with gr.Tabs():
                
                # Chat Tab
                with gr.Tab("💬 Chat with Documentation"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                height=400,
                                label="Programming Assistant",
                                show_label=True,
                                container=True
                            )
                            
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    placeholder="Ask about your documentation... (e.g., 'How do I merge DataFrames in pandas?')",
                                    label="Your Question",
                                    lines=2,
                                    scale=4
                                )
                                send_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                            
                            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        
                        with gr.Column(scale=1):
                            available_docs = list(self.vector_store.get_available_stores().keys())
                            
                            doc_selector = gr.CheckboxGroup(
                                choices=available_docs,
                                value=available_docs,
                                label="📚 Select Documentation Sources",
                                info="Choose which docs to search"
                            )
                            
                            refresh_docs_btn = gr.Button("🔄 Refresh Sources", variant="secondary")
                    
                    # Chat functionality
                    send_btn.click(
                        self.chat_with_docs,
                        inputs=[msg_input, doc_selector, chatbot],
                        outputs=[chatbot, msg_input]
                    )
                    
                    msg_input.submit(
                        self.chat_with_docs,
                        inputs=[msg_input, doc_selector, chatbot],
                        outputs=[chatbot, msg_input]
                    )
                    
                    clear_btn.click(lambda: [], outputs=chatbot)
                    
                    refresh_docs_btn.click(
                        lambda: gr.CheckboxGroup.update(
                            choices=list(self.vector_store.get_available_stores().keys()),
                            value=list(self.vector_store.get_available_stores().keys())
                        ),
                        outputs=doc_selector
                    )
                
                # Add Documentation Tab
                with gr.Tab("📤 Add Documentation"):
                    
                    gr.Markdown("### Add documentation sources to enhance your assistant's knowledge")
                    
                    with gr.Tab("🌐 From URL"):
                        with gr.Row():
                            with gr.Column():
                                url_input = gr.Textbox(
                                    label="Documentation URL",
                                    placeholder="https://docs.python.org/3/library/pandas.html",
                                    info="Supports: Websites, GitHub repos/files, online docs"
                                )
                                url_name_input = gr.Textbox(
                                    label="Name for this documentation",
                                    placeholder="pandas_docs"
                                )
                                add_url_btn = gr.Button("🚀 Add Documentation", variant="primary")
                            
                            with gr.Column():
                                gr.Markdown("""
                                **📋 Supported Sources:**
                                - 📄 Documentation websites
                                - 🐙 GitHub repositories
                                - 📁 GitHub individual files
                                - 📝 Online text/markdown files
                                
                                **Examples:**
                                - `https://docs.python.org/3/library/`
                                - `https://github.com/pandas-dev/pandas`
                                - `https://raw.githubusercontent.com/user/repo/main/README.md`
                                """)
                    
                    with gr.Tab("📁 Upload File"):
                        with gr.Row():
                            with gr.Column():
                                file_input = gr.File(
                                    label="Upload Documentation File",
                                    file_types=[".txt", ".md", ".markdown"],
                                    info="Supported: Text files (.txt), Markdown (.md)"
                                )
                                file_name_input = gr.Textbox(
                                    label="Name for this documentation",
                                    placeholder="my_project_docs"
                                )
                                add_file_btn = gr.Button("📤 Upload Documentation", variant="primary")
                            
                            with gr.Column():
                                gr.Markdown("""
                                **📋 Supported File Types:**
                                - 📝 Text files (.txt)
                                - 📄 Markdown files (.md)
                                
                                **Examples:**
                                - API documentation
                                - Project README files
                                - Technical guides
                                - Code documentation
                                """)
                    
                    # Results area
                    add_result = gr.Textbox(label="Result", interactive=False)
                    
                    # Add documentation handlers
                    add_url_btn.click(
                        self.add_documentation_url,
                        inputs=[url_input, url_name_input],
                        outputs=[add_result, gr.HTML(visible=False)]
                    )
                    
                    add_file_btn.click(
                        self.add_documentation_file,
                        inputs=[file_input, file_name_input],
                        outputs=[add_result, gr.HTML(visible=False)]
                    )
                
                # Manage Documentation Tab
                with gr.Tab("🗂️ Manage Documentation"):
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📚 Available Documentation Sources")
                            docs_display = gr.HTML(value=self._get_available_docs_html())
                            refresh_display_btn = gr.Button("🔄 Refresh List", variant="secondary")
                        
                        with gr.Column():
                            gr.Markdown("### 🗑️ Delete Documentation")
                            available_for_deletion = list(self.vector_store.get_available_stores().keys())
                            delete_selector = gr.Dropdown(
                                choices=available_for_deletion,
                                label="Select documentation to delete",
                                info="Choose a documentation source to remove"
                            )
                            delete_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                            delete_result = gr.Textbox(label="Result", interactive=False)
                    
                    # Management handlers
                    refresh_display_btn.click(
                        lambda: self._get_available_docs_html(),
                        outputs=docs_display
                    )
                    
                    delete_btn.click(
                        self.delete_documentation,
                        inputs=delete_selector,
                        outputs=[delete_result, docs_display]
                    )
                
                # Setup Tab
                with gr.Tab("🔧 Setup & API Keys"):
                    setup_html = gr.HTML(value=self.get_setup_instructions())
                    
                    with gr.Row():
                        check_setup_btn = gr.Button("🔍 Check API Keys", variant="primary")
                        refresh_setup_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    
                    setup_result = gr.Textbox(
                        label="API Key Check Results",
                        lines=10,
                        interactive=False
                    )
                    
                    def check_api_keys():
                        try:
                            from setup_checker import APIKeyChecker
                            checker = APIKeyChecker()
                            results = checker.check_all_keys()
                            
                            # Format results
                            output = "🔑 API Key Status Check:\n\n"
                            for service, info in results.items():
                                output += f"{info['message']}\n"
                                if 'setup_url' in info:
                                    output += f"   Setup: {info['setup_url']}\n"
                                output += "\n"
                            
                            return output
                        except Exception as e:
                            return f"Error checking API keys: {str(e)}"
                    
                    check_setup_btn.click(
                        check_api_keys,
                        outputs=setup_result
                    )
                    
                    refresh_setup_btn.click(
                        lambda: self.get_setup_instructions(),
                        outputs=setup_html
                    )
        
        return interface

def main():
    """Main function to run the application"""
    print("🚀 Starting Programming Helper Agent...")
    
    # Create and run the application
    app = ProgrammingHelperApp()
    interface = app.create_interface()
    
    # Launch with appropriate settings
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()