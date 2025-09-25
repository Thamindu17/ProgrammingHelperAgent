# Programming Helper Agent 🤖

A **RAG-based AI assistant** for programming help, code examples, and technical documentation search. This intelligent agent combines retrieval-augmented generation (RAG) with vector databases to provide contextual, accurate programming assistance.

## ✨ Features

- **🔍 Smart Document Search**: Vector-based similarity search across your documentation
- **💡 Contextual AI Responses**: RAG-powered answers using your own documentation
- **📚 Multi-format Support**: PDF, Markdown, text, code files, and more
- **🧠 Code-aware Processing**: Understands programming languages and preserves code structure
- **🔧 Modular Architecture**: Easy to extend and customize
- **⚡ Interactive CLI**: User-friendly command-line interface
- **📊 Analytics & Stats**: Track usage and knowledge base metrics
- **🔄 Live Updates**: Add new documents without rebuilding

## 🏗️ Architecture

```
programming-helper-agent/
├── 📁 data/              # Documentation files (PDF, MD, TXT, code)
├── 📁 embeddings/        # FAISS index and vector embeddings
├── 📁 agent/             # Core AI agent logic
│   ├── programming_agent.py    # Main orchestrator
│   ├── retriever.py            # Vector search & retrieval
│   └── llm_interface.py        # LLM integration
├── 📁 utils/             # Processing utilities
│   ├── text_processing.py      # Text cleaning & analysis
│   ├── text_chunking.py        # Intelligent text chunking
│   └── file_processing.py      # Multi-format file handling
├── 📁 config/            # Configuration management
│   ├── config.py               # Settings & validation
│   └── .env.example            # Environment template
├── 📄 app.py             # Main application entry point
├── 📄 requirements.txt   # Python dependencies
└── 📄 README.md          # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Git** (for cloning)
- **API Keys** (OpenAI recommended, or use local models)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/programming-helper-agent.git
cd programming-helper-agent

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\\Scripts\\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp config/.env.example config/.env

# Edit configuration (add your API keys)
# Windows:
notepad config/.env
# macOS/Linux:
nano config/.env
```

**Required settings in `.env`:**
```env
# For OpenAI models (recommended)
OPENAI_API_KEY=your_openai_api_key_here

# Or for local models (advanced)
LLM_MODEL=huggingface/microsoft/DialoGPT-medium
```

### 4. Add Documentation

```bash
# Add your programming documentation to the data folder
# Supported formats: PDF, TXT, MD, PY, JS, HTML, CSS, etc.

# Example files you can add:
# - programming_guides.pdf
# - python_documentation.md
# - coding_best_practices.txt
# - your_project_code.py
```

### 5. Initialize Knowledge Base

```bash
# Process documents and build vector embeddings
python app.py --initialize
```

### 6. Start Using the Agent

```bash
# Interactive mode
python app.py

# Or ask a single question
python app.py --query "How do I implement a binary search in Python?"
```

## 💻 Usage Examples

### Interactive Mode

```bash
$ python app.py

🤖 Ask me: What is a Python decorator?

💡 Response:
A Python decorator is a design pattern that allows you to extend or modify 
the behavior of functions or classes without permanently modifying their code...

📚 Based on 3 relevant documents
```

### Available Commands

- `help` - Show available commands
- `stats` - Display knowledge base statistics  
- `search <query>` - Search documents without generating response
- `summary [query]` - Get knowledge base summary
- `add <file_path>` - Add new document to knowledge base
- `export` - Export conversation history
- `reset` - Clear conversation history
- `quit` - Exit application

### Query Prefixes for Better Results

- `code:` - Code-specific questions
- `debug:` - Debugging help
- `explain:` - Concept explanations

**Examples:**
```bash
🤖 Ask me: code: How to implement a binary search in Python?
🤖 Ask me: debug: My function is returning None instead of a value
🤖 Ask me: explain: What is the difference between a list and a tuple?
```

### Command Line Options

```bash
python app.py --help                    # Show help
python app.py --initialize              # Initialize knowledge base
python app.py --query "your question"   # Single query mode
python app.py --stats                   # Show statistics
python app.py --config                  # Show configuration
python app.py --force-rebuild           # Rebuild from scratch
```

## ⚙️ Configuration

### Environment Variables

Create `config/.env` with your settings:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_token_here

# Model Selection
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo

# Text Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# AI Behavior  
TEMPERATURE=0.7
MAX_TOKENS=4000
```

### Model Options

**Recommended LLM Models:**
- `gpt-3.5-turbo` (OpenAI) - Best balance of quality and cost
- `gpt-4` (OpenAI) - Highest quality, more expensive
- `gpt-4-turbo` (OpenAI) - Latest and most capable

**Local/Open Source Options:**
- `microsoft/DialoGPT-medium` - Conversational model
- `codellama/CodeLlama-7b-Python-hf` - Code-specialized
- Any Hugging Face compatible model

**Embedding Models:**
- `sentence-transformers/all-MiniLM-L6-v2` (default) - Fast and good
- `sentence-transformers/all-mpnet-base-v2` - Higher quality
- `text-embedding-ada-002` (OpenAI) - Premium option

## 🛠️ Advanced Usage

### Adding Custom Documents

```python
from agent import ProgrammingHelperAgent

agent = ProgrammingHelperAgent()

# Add a single document
agent.add_document("path/to/your/document.pdf")

# Or batch process a directory
agent.initialize_knowledge_base(force_rebuild=True)
```

### Programmatic Usage

```python
from agent import ProgrammingHelperAgent

# Initialize agent
agent = ProgrammingHelperAgent()

# Ask questions programmatically
response = agent.ask_question("How do I handle exceptions in Python?")

print(response['response'])
print(f"Based on {response['context_count']} documents")

# Search without generating response
results = agent.search_documents("exception handling", top_k=5)
for result in results:
    print(f"Source: {result['source_file']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Similarity: {result['similarity_score']:.3f}")
```

### Custom Configuration

```python
from config import config

# Modify settings programmatically
config.CHUNK_SIZE = 1500
config.TOP_K_RESULTS = 10
config.TEMPERATURE = 0.5

# Validate configuration
validation = config.validate_config()
if not validation['valid']:
    print("Configuration errors:", validation['errors'])
```

## 📊 Monitoring & Analytics

### View Statistics

```bash
python app.py --stats
```

**Sample Output:**
```
=== Programming Helper Agent Status ===
Initialized: True
LLM Model: gpt-3.5-turbo
Documents Indexed: 15
Total Chunks: 342
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Conversations: 23
```

### Export Data

```python
# Export conversation history
agent.export_conversation("my_conversations.json")

# Get detailed statistics
stats = agent.get_stats()
print(f"Knowledge base contains {stats['total_chunks']} chunks")
print(f"From {stats['files_indexed']} files")
```

## 🔧 Troubleshooting

### Common Issues

**1. "No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**2. "OpenAI API key not provided"**
- Add your API key to `config/.env`
- Or use a local model by changing `LLM_MODEL`

**3. "No files found to process"**
- Add documentation files to the `data/` directory
- Check file extensions are supported

**4. "FAISS not available"**
```bash
pip install faiss-cpu
# Or for GPU support:
pip install faiss-gpu
```

**5. "Knowledge base not initialized"**
```bash
python app.py --initialize
```

### Performance Optimization

**For Large Document Collections:**
- Increase `CHUNK_SIZE` to 1500-2000
- Use more powerful embedding models
- Consider using `faiss-gpu` for faster search

**For Better Responses:**
- Use `gpt-4` instead of `gpt-3.5-turbo`
- Increase `TOP_K_RESULTS` to 7-10
- Fine-tune `SIMILARITY_THRESHOLD`

**For Cost Optimization:**
- Use local models (Hugging Face)
- Reduce `MAX_TOKENS` 
- Lower `TOP_K_RESULTS`

## 🧪 Development

### Project Structure Deep Dive

```
agent/
├── programming_agent.py     # Main orchestrator class
├── retriever.py            # Vector store & document retrieval
├── llm_interface.py        # Language model abstraction
└── __init__.py            # Package exports

utils/
├── text_processing.py      # Text cleaning & analysis
├── text_chunking.py        # Smart text segmentation
├── file_processing.py      # Multi-format file readers
└── __init__.py            # Utility exports

config/
├── config.py              # Configuration management
├── .env.example           # Environment template
└── __init__.py           # Config exports
```

### Adding New File Types

```python
# In utils/file_processing.py
def _extract_custom_content(self, file_path: Path) -> Optional[str]:
    """Extract content from custom file format"""
    # Your custom extraction logic here
    pass

# Register the new extension
self.supported_extensions.append('.custom')
```

### Custom LLM Integration

```python
# In agent/llm_interface.py
class CustomLLMInterface(LLMInterface):
    def _initialize_custom_model(self):
        """Initialize your custom model"""
        pass
    
    def _generate_custom_response(self, prompt: str, **kwargs) -> str:
        """Generate response with custom model"""
        pass
```

### Running Tests

```bash
# Test LLM connection
python -c "from agent import test_llm_connection; print(test_llm_connection())"

# Test configuration
python -c "from config import config; print(config.validate_config())"

# Test file processing
python -c "from utils import FileProcessor; fp = FileProcessor(); print(fp.get_file_list('data/'))"
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/programming-helper-agent.git
cd programming-helper-agent

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\\Scripts\\activate on Windows

# Install with development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If you create this

# Run tests
python -m pytest tests/  # If you add tests
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and embeddings
- **Hugging Face** for transformers and sentence-transformers
- **Facebook AI** for FAISS vector search
- **The Python community** for excellent libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/programming-helper-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/programming-helper-agent/discussions)
- **Documentation**: This README and inline code comments

## 🗺️ Roadmap

### Near Term (v1.1)
- [ ] Web interface with Streamlit/Flask
- [ ] Support for more file formats (DOCX, PPTX)
- [ ] Conversation memory across sessions
- [ ] Query suggestion system

### Medium Term (v1.2)  
- [ ] Multi-language support
- [ ] Integration with popular IDEs (VS Code extension)
- [ ] Real-time document monitoring
- [ ] Advanced analytics dashboard

### Long Term (v2.0)
- [ ] Multi-agent collaboration
- [ ] Custom fine-tuning workflows
- [ ] Enterprise deployment options
- [ ] GraphRAG implementation

---

**Happy Coding! 🚀**

*Built with ❤️ for the programming community*