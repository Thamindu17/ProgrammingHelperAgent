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
    
    def chat_with_docs(self, message: str, history: list, selected_sources: list, top_k: int, per_source_limit: int, min_overlap: int):
        """Chat with documentation sources with source filtering and basic relevance filtering."""
        if not message.strip():
            return history, "", selected_sources

        try:
            all_stores = list(self.vector_store.get_available_stores().keys())
            if not all_stores:
                response = "❌ No documentation available. Please add some documentation first!"
            else:
                if not selected_sources:
                    # default to all
                    selected_sources = all_stores
                # Retrieve
                raw_docs = self.vector_store.search_multiple_stores(
                    message,
                    store_names=selected_sources,
                    k=top_k
                )
                # Group & limit per source
                grouped = {}
                for d in raw_docs:
                    src = d.metadata.get('vector_store','unknown')
                    grouped.setdefault(src,[]).append(d)
                limited_docs = []
                for src, docs in grouped.items():
                    limited_docs.extend(docs[:per_source_limit])

                # Simple token overlap filter
                query_terms = {t.lower() for t in message.split() if len(t) > 3}
                filtered = []
                for d in limited_docs:
                    content_terms = set(w.lower() for w in d.page_content.split())
                    overlap = len(query_terms & content_terms)
                    if overlap >= min_overlap or d.metadata.get('fallback') == 'keyword':
                        filtered.append(d)

                final_docs = filtered if filtered else limited_docs
                if final_docs:
                    response = self.qa_agent.generate_response(message, final_docs)
                else:
                    response = "❌ No relevant information after filtering. Adjust overlap threshold or sources."

            history.append([message, response])
        except Exception as e:
            print(f"Error in chat: {e}")
            response = f"❌ Error: {str(e)}"
            history.append([message, response])
        return history, "", selected_sources
    
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

    # New helper wrappers for UI
    def list_store_names(self):
        return list(self.vector_store.get_available_stores().keys()) or ["(none)"]

    def inspect_store(self, store_name: str):
        if store_name == "(none)":
            return "No stores available"
        samples = self.vector_store.inspect_store_chunks(store_name, limit=3)
        if not samples:
            return f"No chunks found for '{store_name}'"
        out = [f"🔍 First {len(samples)} chunk previews for {store_name}:"]
        for i, s in enumerate(samples, 1):
            meta = {k: v for k, v in s['metadata'].items() if k not in ('source',)}
            out.append(f"\n[{i}] {s['preview'][:180]}...")
            out.append(f"   Meta: {meta}")
        return "\n".join(out)

    def get_system_status(self):
        try:
            from config.settings import feature_summary
            vec_stats = self.vector_store.get_statistics()
            qa_status = self.qa_agent.get_status()
            status_lines = [
                "⚙️ System Status",
                feature_summary(),
                f"Vector Stores: {vec_stats['total_stores']} | Docs: {vec_stats['total_documents']}",
                f"Mode: {qa_status['response_mode']} | Minimal: {qa_status['minimal_mode']}",
            ]
            return "\n".join(status_lines)
        except Exception as e:
            return f"Status error: {e}"

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
            # Explicit type to suppress deprecation warning (tuples retained for backward compat)
            chatbot = gr.Chatbot(height=400, type="tuples")
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask about your documentation...", scale=4)
                send_btn = gr.Button("Ask", scale=1)
            with gr.Accordion("Retrieval Options", open=False):
                source_select = gr.CheckboxGroup(
                    label="Select Sources (empty = all)",
                    choices=app.vector_store.get_available_stores().keys()
                )
                top_k = gr.Slider(1, 10, value=4, step=1, label="Top K per Query")
                per_source = gr.Slider(1, 5, value=2, step=1, label="Per-Source Limit")
                min_overlap = gr.Slider(0, 10, value=1, step=1, label="Min Token Overlap")
            clear = gr.Button("Clear Chat")

            def _chat_wrapper(message, history, selected, k, per_src, overlap):
                return app.chat_with_docs(message, history, selected, int(k), int(per_src), int(overlap))

            send_btn.click(
                _chat_wrapper,
                inputs=[msg, chatbot, source_select, top_k, per_source, min_overlap],
                outputs=[chatbot, msg, source_select]
            )
            msg.submit(
                _chat_wrapper,
                inputs=[msg, chatbot, source_select, top_k, per_source, min_overlap],
                outputs=[chatbot, msg, source_select]
            )
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

            gr.Markdown("### Inspect Chunks")
            store_dropdown = gr.Dropdown(
                label="Select Store",
                choices=app.list_store_names(),
                value=app.list_store_names()[0]
            )
            inspect_btn = gr.Button("👓 Preview Chunks")
            inspect_output = gr.Textbox(label="Chunk Preview", lines=8)
            def refresh_choices():
                return gr.Dropdown(choices=app.list_store_names(), value=app.list_store_names()[0])
            # Update dropdown on refresh
            refresh_btn.click(lambda: None, None, None, js="()=>{}")  # No-op to keep existing behavior
            inspect_btn.click(app.inspect_store, inputs=store_dropdown, outputs=inspect_output)

        with gr.Tab("🧪 Status"):
            status_box = gr.Textbox(label="System Status", value=app.get_system_status(), lines=6)
            status_refresh = gr.Button("🔄 Refresh Status")
            status_refresh.click(app.get_system_status, outputs=status_box)
        
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
    
    # Determine port: env GRADIO_SERVER_PORT > fallback to first free in list
    import socket
    def find_free_port(candidates=(7863, 7865, 7866, 8000, 9000)):
        for p in candidates:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        # Let OS choose
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    import os as _os
    env_port = _os.getenv("GRADIO_SERVER_PORT")
    if env_port and env_port.isdigit():
        port = int(env_port)
    else:
        port = find_free_port()
    print(f"🌐 Launching on http://127.0.0.1:{port} (set GRADIO_SERVER_PORT to override)")

    interface.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False
    )

if __name__ == "__main__":
    main()