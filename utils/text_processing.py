"""
Text processing utilities for the Programming Helper Agent
Handles text cleaning, normalization, and preprocessing
"""

import re
import string
from typing import List, Optional
import unicodedata


class TextCleaner:
    """Utility class for cleaning and preprocessing text"""
    
    def __init__(self):
        self.programming_keywords = {
            'python', 'javascript', 'java', 'cpp', 'c++', 'html', 'css', 
            'sql', 'git', 'api', 'json', 'xml', 'http', 'https'
        }
    
    def clean_text(self, text: str, preserve_code: bool = True) -> str:
        """
        Clean and normalize text while optionally preserving code formatting
        
        Args:
            text: Raw text to clean
            preserve_code: Whether to preserve code blocks and formatting
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace special characters but preserve programming syntax
        if preserve_code:
            # Preserve important programming characters
            text = self._preserve_code_syntax(text)
        else:
            # Standard text cleaning
            text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)]', ' ', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        
        return text.strip()
    
    def _preserve_code_syntax(self, text: str) -> str:
        """Preserve important programming syntax characters"""
        # Don't remove these characters as they're important for code
        code_chars = r'[{}()[\];:.,<>=+\-*/%&|!^~@#$]'
        
        # Only remove truly problematic characters
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\{\}\[\]<>=+\-\*/\%\&\|\^\~@#\$\'\"\\\/]', ' ', text)
        
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove excessive whitespace while preserving structure"""
        # Remove trailing whitespace from lines
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Remove excessive blank lines (more than 2 consecutive)
        cleaned_lines = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_code_blocks(self, text: str) -> List[dict]:
        """
        Extract code blocks from markdown-style text
        
        Returns:
            List of dictionaries with 'language', 'code', and 'line_start'
        """
        code_blocks = []
        
        # Pattern for fenced code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or 'unknown'
            code = match.group(2)
            line_start = text[:match.start()].count('\n')
            
            code_blocks.append({
                'language': language.lower(),
                'code': code,
                'line_start': line_start,
                'original_match': match.group(0)
            })
        
        return code_blocks
    
    def is_likely_code(self, text: str) -> bool:
        """
        Determine if text is likely to be code based on patterns
        
        Args:
            text: Text to analyze
            
        Returns:
            Boolean indicating if text appears to be code
        """
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python function definition
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statement
            r'#include\s*<',  # C/C++ include
            r'<\w+[^>]*>',  # HTML tags
            r'\{\s*[\w\-]+\s*:',  # CSS/JSON objects
            r'SELECT\s+.*FROM',  # SQL
            r'git\s+\w+',  # Git commands
        ]
        
        # Count matches
        matches = sum(1 for pattern in code_indicators if re.search(pattern, text, re.IGNORECASE))
        
        # Also check for high concentration of programming keywords
        words = text.lower().split()
        keyword_count = sum(1 for word in words if word in self.programming_keywords)
        keyword_ratio = keyword_count / len(words) if words else 0
        
        return matches >= 2 or keyword_ratio > 0.1
    
    def preserve_structure(self, text: str) -> str:
        """
        Preserve important document structure like headers, lists, etc.
        
        Args:
            text: Text with potential structure
            
        Returns:
            Text with preserved structure markers
        """
        # Preserve markdown headers
        text = re.sub(r'^(#{1,6})\s+(.+)$', r'\1 \2', text, flags=re.MULTILINE)
        
        # Preserve list items
        text = re.sub(r'^(\s*[\-\*\+])\s+(.+)$', r'\1 \2', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*\d+\.)\s+(.+)$', r'\1 \2', text, flags=re.MULTILINE)
        
        return text


def clean_filename(filename: str) -> str:
    """
    Clean filename for safe storage and processing
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove path separators and dangerous characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove excessive dots (but keep file extension)
    name_parts = cleaned.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        name = re.sub(r'\.+', '_', name)
        cleaned = f"{name}.{ext}"
    
    return cleaned.strip()


def detect_programming_language(text: str) -> Optional[str]:
    """
    Detect the programming language of a code snippet
    
    Args:
        text: Code text to analyze
        
    Returns:
        Detected language name or None if uncertain
    """
    # Language patterns (order matters - more specific first)
    patterns = {
        'python': [
            r'def\s+\w+\s*\(',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'if\s+__name__\s*==\s*["\']__main__["\']',
            r'print\s*\(',
        ],
        'javascript': [
            r'function\s+\w+\s*\(',
            r'const\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'console\.log\s*\(',
            r'=>\s*{',
        ],
        'java': [
            r'public\s+class\s+\w+',
            r'public\s+static\s+void\s+main',
            r'System\.out\.print',
            r'import\s+java\.',
        ],
        'cpp': [
            r'#include\s*<\w+>',
            r'int\s+main\s*\(',
            r'std::\w+',
            r'cout\s*<<',
        ],
        'html': [
            r'<html[^>]*>',
            r'<head[^>]*>',
            r'<body[^>]*>',
            r'<div[^>]*>',
        ],
        'css': [
            r'\w+\s*{\s*[\w\-]+\s*:',
            r'@media\s+',
            r'#\w+\s*{',
            r'\.\w+\s*{',
        ],
        'sql': [
            r'SELECT\s+.*FROM',
            r'INSERT\s+INTO',
            r'UPDATE\s+.*SET',
            r'CREATE\s+TABLE',
        ]
    }
    
    text_lower = text.lower()
    
    for language, lang_patterns in patterns.items():
        matches = sum(1 for pattern in lang_patterns 
                     if re.search(pattern, text_lower, re.IGNORECASE))
        if matches >= 2:
            return language
    
    return None