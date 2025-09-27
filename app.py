
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent import ProgrammingHelperAgent, test_llm_connection
from config import config


def print_banner():
    """Print the application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 Programming Helper Agent                     ║
    ║                  RAG-based AI Assistant                      ║
    ║                                                              ║
    ║  Your intelligent companion for programming questions,       ║
    ║  code examples, and technical documentation search           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print usage help"""
    help_text = """
Available Commands:
    help, h, ?           - Show this help message
    stats, status        - Show agent statistics
    search <query>       - Search documents without generating response
    summary [query]      - Get summary of knowledge base
    reset                - Reset conversation history
    export               - Export conversation to file
    config               - Show current configuration
    reload               - Reload knowledge base
    rebuild              - Rebuild knowledge base from scratch
    add <file_path>      - Add a new document to knowledge base
    quit, exit, q        - Exit the application

Special Modes:
    Use 'code:' prefix for code-specific questions
    Use 'debug:' prefix for debugging help
    Use 'explain:' prefix for concept explanations

Examples:
    > What is a Python decorator?
    > code: How to implement a binary search in Python?
    > debug: My function is returning None instead of a value
    > explain: What is the difference between a list and a tuple?
    """
    print(help_text)


def validate_environment() -> Dict[str, Any]:
    """Validate the environment and configuration"""
    print("Validating environment...")
    
    validation_results = {
        'config_valid': True,
        'llm_available': False,
        'data_dir_exists': False,
        'warnings': [],
        'errors': []
    }
    
    # Validate configuration
    config_validation = config.validate_config()
    validation_results['config_valid'] = config_validation['valid']
    validation_results['warnings'].extend(config_validation.get('warnings', []))
    validation_results['errors'].extend(config_validation.get('errors', []))
    
    # Check LLM availability
    llm_test = test_llm_connection()
    validation_results['llm_available'] = llm_test['success']
    if not llm_test['success']:
        validation_results['warnings'].append(f"LLM not available: {llm_test.get('error', 'Unknown error')}")
    
    # Check data directory
    data_dir = Path(config.DATA_DIR)
    validation_results['data_dir_exists'] = data_dir.exists()
    if not data_dir.exists():
        validation_results['warnings'].append(f"Data directory {data_dir} does not exist")
    
    # Print validation results
    if validation_results['errors']:
        print("❌ Validation Errors:")
        for error in validation_results['errors']:
            print(f"   • {error}")
    
    if validation_results['warnings']:
        print("⚠️  Warnings:")
        for warning in validation_results['warnings']:
            print(f"   • {warning}")
    
    if not validation_results['errors']:
        print("✅ Environment validation completed")
    
    return validation_results


def initialize_agent(force_rebuild: bool = False) -> Optional[ProgrammingHelperAgent]:
    """Initialize the Programming Helper Agent"""
    try:
        print("\n🚀 Initializing Programming Helper Agent...")
        
        # Validate environment first
        validation = validate_environment()
        
        if validation['errors']:
            print("\n❌ Cannot initialize agent due to configuration errors.")
            print("Please fix the errors above and try again.")
            return None
        
        # Create agent
        agent = ProgrammingHelperAgent(load_existing=not force_rebuild)
        
        # Initialize knowledge base if needed
        if not agent.is_initialized or force_rebuild:
            if not agent.data_dir.exists():
                print(f"\n📁 Creating data directory: {agent.data_dir}")
                agent.data_dir.mkdir(parents=True, exist_ok=True)
                print("\n📋 Please add your documentation files to the data directory:")
                print(f"   {agent.data_dir.absolute()}")
                print("\nSupported formats: PDF, TXT, MD, PY, JS, HTML, CSS, and more")
                print("Then run: python app.py --initialize")
                return agent
            
            print("\n🔄 Initializing knowledge base...")
            if agent.initialize_knowledge_base(force_rebuild=force_rebuild):
                print("✅ Knowledge base initialized successfully!")
            else:
                print("❌ Failed to initialize knowledge base")
                return None
        
        return agent
        
    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        return None


def interactive_mode(agent: ProgrammingHelperAgent):
    """Run the agent in interactive mode"""
    print("\n🎯 Interactive Mode - Ask me anything about programming!")
    print("Type 'help' for commands, 'quit' to exit")
    
    if not agent.is_initialized:
        print("\n⚠️  Knowledge base not initialized. Some features may be limited.")
        print("Add documents to the data directory and run --initialize")
    
    print("\n" + "="*60)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 Ask me: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Happy coding!")
                break
            
            elif user_input.lower() in ['help', 'h', '?']:
                print_help()
                continue
            
            elif user_input.lower() in ['stats', 'status']:
                agent.print_status()
                continue
            
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    print(f"\n🔍 Searching for: {query}")
                    results = agent.search_documents(query)
                    if results:
                        print(f"\nFound {len(results)} relevant documents:")
                        for i, result in enumerate(results[:5], 1):
                            print(f"\n{i}. {result['source_file']} (Score: {result['similarity_score']:.3f})")
                            print(f"   {result['content'][:200]}...")
                    else:
                        print("No relevant documents found.")
                continue
            
            elif user_input.lower().startswith('summary'):
                query = user_input[7:].strip() if len(user_input) > 7 else None
                print(f"\n📄 Generating summary...")
                summary = agent.get_document_summary(query)
                print(f"\n{summary}")
                continue
            
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                continue
            
            elif user_input.lower() == 'export':
                file_path = agent.export_conversation()
                if file_path:
                    print(f"✅ Conversation exported to: {file_path}")
                continue
            
            elif user_input.lower() == 'config':
                config.print_config()
                continue
            
            elif user_input.lower() == 'reload':
                print("🔄 Reloading knowledge base...")
                if agent.load_knowledge_base():
                    print("✅ Knowledge base reloaded")
                else:
                    print("❌ Failed to reload knowledge base")
                continue
            
            elif user_input.lower() == 'rebuild':
                print("🔄 Rebuilding knowledge base from scratch...")
                if agent.initialize_knowledge_base(force_rebuild=True):
                    print("✅ Knowledge base rebuilt successfully")
                else:
                    print("❌ Failed to rebuild knowledge base")
                continue
            
            elif user_input.lower().startswith('add '):
                file_path = user_input[4:].strip()
                if file_path:
                    if agent.add_document(file_path):
                        print(f"✅ Added document: {file_path}")
                    else:
                        print(f"❌ Failed to add document: {file_path}")
                continue
            
            # Regular question
            print(f"\n🤔 Thinking about: {user_input}")
            print("⏳ Searching knowledge base and generating response...")
            
            response_data = agent.ask_question(user_input)
            
            if response_data['success']:
                print(f"\n💡 Response:")
                print("-" * 50)
                print(response_data['response'])
                
                if response_data.get('context_count', 0) > 0:
                    print(f"\n📚 Based on {response_data['context_count']} relevant documents")
                
            else:
                print(f"\n❌ Error: {response_data['response']}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy coding!")
            break
        except EOFError:
            print("\n\n👋 Goodbye! Happy coding!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


def single_query_mode(agent: ProgrammingHelperAgent, query: str):
    """Process a single query and exit"""
    print(f"\n🤔 Processing query: {query}")
    
    response_data = agent.ask_question(query)
    
    if response_data['success']:
        print(f"\n💡 Response:")
        print("=" * 60)
        print(response_data['response'])
        
        if response_data.get('context_count', 0) > 0:
            print(f"\n📚 Based on {response_data['context_count']} relevant documents")
        
        # Also print the sources if any
        if response_data.get('context'):
            print(f"\n📖 Sources:")
            sources = set()
            for ctx in response_data['context'][:3]:
                sources.add(ctx.get('source_file', 'Unknown'))
            for source in sorted(sources):
                print(f"   • {source}")
    else:
        print(f"\n❌ Error: {response_data['response']}")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Programming Helper Agent - RAG-based AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py                               # Interactive mode
    python app.py --initialize                  # Initialize knowledge base
    python app.py --query "What is recursion?" # Single query
    python app.py --stats                       # Show statistics
        """
    )
    
    parser.add_argument(
        '--initialize', '-i',
        action='store_true',
        help='Initialize or rebuild the knowledge base'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Ask a single question and exit'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show agent statistics and exit'
    )
    
    parser.add_argument(
        '--config', '-c',
        action='store_true',
        help='Show configuration and exit'
    )
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of knowledge base (use with --initialize)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Override data directory path'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle config display
    if args.config:
        config.print_config()
        return
    
    # Override data directory if specified
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    
    # Initialize agent
    agent = initialize_agent(force_rebuild=args.force_rebuild)
    
    if not agent:
        sys.exit(1)
    
    # Handle different modes
    if args.stats:
        agent.print_status()
        return
    
    elif args.initialize:
        if agent.initialize_knowledge_base(force_rebuild=True):
            print("\n✅ Knowledge base initialization completed!")
            agent.print_status()
        else:
            print("\n❌ Failed to initialize knowledge base")
            sys.exit(1)
        return
    
    elif args.query:
        single_query_mode(agent, args.query)
        return
    
    else:
        # Interactive mode
        interactive_mode(agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Happy coding!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)