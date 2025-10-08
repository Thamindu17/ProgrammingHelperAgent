# agent/enhanced_agent.py

from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from agent.main_agent import create_qa_chain
from services.code_executor import CodeExecutor
from services.stackoverflow_search import StackOverflowSearcher
from services.github_search import GitHubSearcher
from config.settings import *

class EnhancedProgrammingAgent:
    """
    Enhanced Programming Helper Agent with free API integrations
    """
    
    def __init__(self):
        # Initialize core components
        self.qa_chain = create_qa_chain()
        self.code_executor = CodeExecutor()
        self.stackoverflow_searcher = StackOverflowSearcher()
        self.github_searcher = GitHubSearcher()
        
        # Agent capabilities
        self.capabilities = {
            "code_execution": ENABLE_CODE_EXECUTION,
            "stackoverflow_search": ENABLE_STACKOVERFLOW_SEARCH,
            "github_search": ENABLE_GITHUB_SEARCH,
            "translation": ENABLE_TRANSLATION
        }
    
    def process_query(self, user_query: str, context: Dict = None) -> Dict:
        """
        Process user query with enhanced capabilities
        
        Args:
            user_query: User's programming question
            context: Additional context (language, preferences, etc.)
            
        Returns:
            Comprehensive response with multiple sources
        """
        context = context or {}
        programming_language = context.get('language', self._detect_language(user_query))
        
        # Step 1: Get base answer from RAG system
        base_response = self._get_base_answer(user_query)
        
        # Step 2: Search for additional context
        additional_context = self._gather_additional_context(user_query, programming_language)
        
        # Step 3: Detect if user wants code execution
        execution_result = None
        if self._wants_code_execution(user_query):
            code = self._extract_code_from_query(user_query)
            if code:
                execution_result = self.code_executor.execute_code(code, programming_language)
        
        # Step 4: Compile comprehensive response
        return self._compile_response(
            user_query=user_query,
            base_response=base_response,
            additional_context=additional_context,
            execution_result=execution_result,
            programming_language=programming_language
        )
    
    def _get_base_answer(self, query: str) -> Dict:
        """Get answer from the base RAG system"""
        try:
            result = self.qa_chain.invoke({"query": query})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Error getting base answer: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def _gather_additional_context(self, query: str, language: str) -> Dict:
        """Gather additional context from free APIs"""
        context = {
            "stackoverflow": None,
            "github": None,
            "api_status": {}
        }
        
        # Stack Overflow search
        if self.capabilities["stackoverflow_search"]:
            try:
                so_results = self.stackoverflow_searcher.search_with_context(query, language)
                context["stackoverflow"] = so_results
                context["api_status"]["stackoverflow"] = "success"
            except Exception as e:
                context["api_status"]["stackoverflow"] = f"error: {str(e)}"
        
        # GitHub search
        if self.capabilities["github_search"]:
            try:
                gh_results = self.github_searcher.get_popular_examples(language, query)
                context["github"] = gh_results
                context["api_status"]["github"] = "success"
            except Exception as e:
                context["api_status"]["github"] = f"error: {str(e)}"
        
        return context
    
    def _detect_language(self, query: str) -> str:
        """Simple language detection based on keywords"""
        language_keywords = {
            'python': ['python', 'pandas', 'numpy', 'django', 'flask', 'jupyter'],
            'javascript': ['javascript', 'js', 'react', 'node', 'vue', 'angular'],
            'java': ['java', 'spring', 'maven', 'gradle', 'android'],
            'cpp': ['c++', 'cpp', 'stl', 'boost'],
            'c': ['c programming', 'malloc', 'pointer'],
            'csharp': ['c#', 'csharp', '.net', 'asp.net'],
            'go': ['golang', 'go programming'],
            'rust': ['rust programming', 'cargo'],
            'php': ['php', 'laravel', 'wordpress'],
            'ruby': ['ruby', 'rails']
        }
        
        query_lower = query.lower()
        for lang, keywords in language_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return lang
        
        return 'python'  # Default to Python
    
    def _wants_code_execution(self, query: str) -> bool:
        """Detect if user wants to execute code"""
        execution_keywords = [
            'run this code', 'execute', 'what does this output',
            'test this', 'try this code', 'run it', 'execute this'
        ]
        return any(keyword in query.lower() for keyword in execution_keywords)
    
    def _extract_code_from_query(self, query: str) -> Optional[str]:
        """Extract code from user query"""
        # Simple code extraction - look for code blocks
        lines = query.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if '```' in line:
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # If no code blocks, look for indented code
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line.strip())
        
        return '\n'.join(code_lines) if code_lines else None
    
    def _compile_response(self, user_query: str, base_response: Dict, 
                         additional_context: Dict, execution_result: Dict = None,
                         programming_language: str = None) -> Dict:
        """Compile comprehensive response"""
        response = {
            "main_answer": base_response["answer"],
            "programming_language": programming_language,
            "sources": {
                "knowledge_base": bool(base_response["success"]),
                "stackoverflow": bool(additional_context.get("stackoverflow", {}).get("found")),
                "github": bool(additional_context.get("github", {}).get("code_examples")),
                "code_execution": bool(execution_result and execution_result.get("status") != "error")
            },
            "additional_resources": {},
            "code_execution": execution_result,
            "api_status": additional_context.get("api_status", {}),
            "capabilities_used": [cap for cap, enabled in self.capabilities.items() if enabled]
        }
        
        # Add Stack Overflow results
        if additional_context.get("stackoverflow", {}).get("found"):
            so_data = additional_context["stackoverflow"]
            response["additional_resources"]["stackoverflow"] = {
                "top_question": so_data["questions"][0] if so_data["questions"] else None,
                "total_questions": so_data.get("total_questions", 0)
            }
        
        # Add GitHub results
        if additional_context.get("github", {}).get("code_examples"):
            gh_data = additional_context["github"]
            response["additional_resources"]["github"] = {
                "code_examples": gh_data["code_examples"][:2],  # Top 2 examples
                "repositories": gh_data["related_repositories"][:2]  # Top 2 repos
            }
        
        return response
    
    def get_capabilities_status(self) -> Dict:
        """Get status of all capabilities"""
        return {
            "enabled_features": [cap for cap, enabled in self.capabilities.items() if enabled],
            "disabled_features": [cap for cap, enabled in self.capabilities.items() if not enabled],
            "api_keys_configured": {
                "judge0": bool(JUDGE0_API_KEY),
                "github": bool(GITHUB_TOKEN),
                "stackoverflow": bool(STACKOVERFLOW_KEY),
                "google_translate": bool(GOOGLE_TRANSLATE_KEY)
            },
            "recommendations": self._get_setup_recommendations()
        }
    
    def _get_setup_recommendations(self) -> List[str]:
        """Get recommendations for setting up free APIs"""
        recommendations = []
        
        if not JUDGE0_API_KEY:
            recommendations.append("Add JUDGE0_API_KEY to enable code execution (free: 50 requests/day)")
        
        if not GITHUB_TOKEN:
            recommendations.append("Add GITHUB_TOKEN to enable GitHub code search (free: 5000 requests/hour)")
        
        if not STACKOVERFLOW_KEY:
            recommendations.append("Add STACKOVERFLOW_KEY to enhance Stack Overflow search (free: 10k requests/day)")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    agent = EnhancedProgrammingAgent()
    
    # Test query
    query = "How do I merge two pandas DataFrames in Python?"
    result = agent.process_query(query)
    
    print("Enhanced Agent Response:")
    print(f"Main Answer: {result['main_answer'][:200]}...")
    print(f"Sources Used: {result['sources']}")
    print(f"Capabilities: {result['capabilities_used']}")
    
    # Check capabilities
    status = agent.get_capabilities_status()
    print(f"\nCapabilities Status: {status}")