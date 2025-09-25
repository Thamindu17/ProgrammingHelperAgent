"""
Language model interface and response generation for Programming Helper Agent
Handles interaction with different LLMs (OpenAI, Hugging Face, local models)
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Import for OpenAI (will need to be installed)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import for Hugging Face transformers (will need to be installed)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import config


class LLMInterface:
    """Interface for interacting with different language models"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the LLM interface
        
        Args:
            model_name: Name of the model to use
            api_key: API key for external services
        """
        self.model_name = model_name or config.LLM_MODEL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Initialize based on model type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate model based on model_name"""
        if self.model_name.startswith('gpt-'):
            self._initialize_openai()
        elif self.model_name.startswith('huggingface/'):
            self._initialize_huggingface()
        else:
            # Assume it's a Hugging Face model
            self._initialize_huggingface()
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            print("OpenAI package not available. Install with: pip install openai")
            return
        
        if not self.api_key:
            print("OpenAI API key not provided")
            return
        
        try:
            openai.api_key = self.api_key
            print(f"Initialized OpenAI with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing OpenAI: {str(e)}")
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers package not available. Install with: pip install transformers")
            return
        
        try:
            # Remove 'huggingface/' prefix if present
            model_name = self.model_name.replace('huggingface/', '')
            
            print(f"Loading Hugging Face model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=config.MAX_TOKENS
            )
            
            print(f"Initialized Hugging Face model: {model_name}")
            
        except Exception as e:
            print(f"Error initializing Hugging Face model: {str(e)}")
    
    def generate_response(self, prompt: str, context: List[Dict[str, Any]] = None,
                         max_tokens: int = None, temperature: float = None) -> str:
        """
        Generate a response using the language model
        
        Args:
            prompt: The user's question or prompt
            context: List of relevant document chunks for context
            max_tokens: Maximum tokens in response
            temperature: Randomness in generation (0.0 to 1.0)
            
        Returns:
            Generated response string
        """
        max_tokens = max_tokens or config.MAX_TOKENS
        temperature = temperature or config.TEMPERATURE
        
        # Build the full prompt with context
        full_prompt = self._build_prompt_with_context(prompt, context)
        
        try:
            if self.model_name.startswith('gpt-'):
                return self._generate_openai_response(full_prompt, max_tokens, temperature)
            else:
                return self._generate_huggingface_response(full_prompt, max_tokens, temperature)
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return f"I apologize, but I encountered an error while generating a response. Please try again or check your configuration.\n\nError: {error_msg}"
    
    def _build_prompt_with_context(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Build a comprehensive prompt including retrieved context"""
        
        # System message for the programming helper
        system_prompt = """You are a Programming Helper Agent, an expert assistant specializing in programming, software development, and technical documentation. Your role is to:

1. Answer programming questions clearly and accurately
2. Provide code examples when helpful
3. Explain concepts in a beginner-friendly way when needed
4. Reference provided documentation and code examples
5. Suggest best practices and improvements
6. Help debug and troubleshoot issues

When answering:
- Be concise but thorough
- Use code examples to illustrate points
- Reference the provided context when relevant
- If you're unsure about something, say so
- Format code clearly with appropriate syntax highlighting hints"""

        # Add context from retrieved documents
        context_section = ""
        if context:
            context_section = "\n\nRelevant Documentation and Code Examples:\n"
            context_section += "=" * 50 + "\n"
            
            for i, ctx in enumerate(context[:5], 1):  # Limit to top 5 results
                source = ctx.get('source_file', 'Unknown')
                content = ctx.get('content', '')
                similarity = ctx.get('similarity_score', 0)
                
                context_section += f"\n[Context {i}] From: {source} (Relevance: {similarity:.2f})\n"
                context_section += "-" * 30 + "\n"
                context_section += content[:1000]  # Limit context length
                if len(content) > 1000:
                    context_section += "...\n"
                context_section += "\n"
            
            context_section += "=" * 50 + "\n"
        
        # Build final prompt
        full_prompt = f"""{system_prompt}

{context_section}

User Question: {prompt}

Please provide a helpful response based on the context above and your knowledge:"""
        
        return full_prompt
    
    def _generate_openai_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using OpenAI API"""
        if not OPENAI_AVAILABLE or not self.api_key:
            return "OpenAI not available or API key not provided."
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    def _generate_huggingface_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Hugging Face model"""
        if not self.pipeline:
            return "Hugging Face model not available."
        
        try:
            # Generate response
            generated = self.pipeline(
                prompt,
                max_length=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text (remove the original prompt)
            full_text = generated[0]['generated_text']
            response = full_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Error with Hugging Face model: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if the LLM is properly initialized and available"""
        if self.model_name.startswith('gpt-'):
            return OPENAI_AVAILABLE and bool(self.api_key)
        else:
            return bool(self.pipeline)


class ResponseSummarizer:
    """Utility class for summarizing and formatting responses"""
    
    def __init__(self, llm_interface: LLMInterface = None):
        """
        Initialize the response summarizer
        
        Args:
            llm_interface: LLMInterface instance to use
        """
        self.llm = llm_interface or LLMInterface()
    
    def summarize_documents(self, documents: List[Dict[str, Any]], 
                           query: str = None, max_length: int = 500) -> str:
        """
        Create a summary of multiple documents
        
        Args:
            documents: List of document chunks
            query: Original query for context
            max_length: Maximum length of summary
            
        Returns:
            Summary string
        """
        if not documents:
            return "No relevant documents found."
        
        # Combine document contents
        combined_content = ""
        sources = set()
        
        for doc in documents[:3]:  # Limit to top 3 documents
            content = doc.get('content', '')
            source = doc.get('source_file', 'Unknown')
            sources.add(source)
            
            combined_content += f"\n\nFrom {source}:\n{content[:800]}"
        
        # Create summary prompt
        summary_prompt = f"""Please provide a concise summary of the following programming documentation and code examples:

{combined_content}

Focus on:
- Key concepts and main points
- Important code patterns or examples
- How this relates to: {query if query else 'programming questions'}

Summary (max {max_length} characters):"""

        try:
            summary = self.llm.generate_response(
                summary_prompt,
                max_tokens=max_length // 4,  # Rough token estimation
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            # Add source information
            source_list = ", ".join(sorted(sources))
            summary += f"\n\nSources: {source_list}"
            
            return summary
            
        except Exception as e:
            print(f"Error creating summary: {str(e)}")
            return f"Found {len(documents)} relevant documents from {len(sources)} sources."
    
    def format_code_response(self, response: str, language: str = None) -> str:
        """
        Format a response that contains code examples
        
        Args:
            response: Raw response string
            language: Programming language for syntax highlighting
            
        Returns:
            Formatted response with proper code blocks
        """
        # This is a simple implementation
        # In a full version, you might want more sophisticated parsing
        
        if not language:
            # Try to detect language from response
            if 'python' in response.lower() or 'def ' in response:
                language = 'python'
            elif 'javascript' in response.lower() or 'function(' in response:
                language = 'javascript'
            elif 'java' in response.lower() or 'public class' in response:
                language = 'java'
        
        # Add code block formatting if not already present
        if '```' not in response and language:
            # Look for code-like patterns and wrap them
            import re
            
            # Simple pattern matching for code blocks
            code_patterns = [
                r'(\n[ ]{4,}.*\n)',  # Indented code
                r'(def .*?:\n.*?(?=\n\S|\Z))',  # Python functions
                r'(function .*?\{.*?\})',  # JavaScript functions
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    formatted_code = f"```{language}\n{match.strip()}\n```"
                    response = response.replace(match, f"\n{formatted_code}\n")
        
        return response


def test_llm_connection(model_name: str = None) -> Dict[str, Any]:
    """
    Test connection to the specified language model
    
    Args:
        model_name: Model to test (defaults to config model)
        
    Returns:
        Dictionary with test results
    """
    model_name = model_name or config.LLM_MODEL
    
    try:
        llm = LLMInterface(model_name)
        
        if not llm.is_available():
            return {
                'success': False,
                'model': model_name,
                'error': 'Model not available or not properly configured'
            }
        
        # Simple test query
        test_response = llm.generate_response(
            "What is Python?",
            max_tokens=100,
            temperature=0.1
        )
        
        return {
            'success': True,
            'model': model_name,
            'test_response_length': len(test_response),
            'test_response_preview': test_response[:200] + "..." if len(test_response) > 200 else test_response
        }
        
    except Exception as e:
        return {
            'success': False,
            'model': model_name,
            'error': str(e)
        }