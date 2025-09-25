"""
Text chunking utilities for the Programming Helper Agent
Handles splitting documents into manageable chunks for embedding and retrieval
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Data class representing a text chunk with metadata"""
    content: str
    chunk_id: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    language: Optional[str] = None
    is_code: bool = False


class TextChunker:
    """Utility class for chunking text documents intelligently"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = 100  # Minimum viable chunk size
    
    def chunk_text(self, text: str, source_file: str = "", 
                   preserve_structure: bool = True) -> List[TextChunk]:
        """
        Split text into chunks with intelligent boundary detection
        
        Args:
            text: Text to chunk
            source_file: Source filename for metadata
            preserve_structure: Whether to preserve document structure
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text) < self.min_chunk_size:
            return []
        
        chunks = []
        
        if preserve_structure:
            # Try structure-aware chunking first
            chunks = self._structure_aware_chunk(text, source_file)
        
        if not chunks:
            # Fall back to sliding window chunking
            chunks = self._sliding_window_chunk(text, source_file)
        
        return chunks
    
    def _structure_aware_chunk(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Chunk text while preserving document structure (headers, code blocks, etc.)
        """
        chunks = []
        
        # Split by major structure elements
        sections = self._split_by_structure(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for section in sections:
            section_content = section['content']
            section_type = section['type']
            
            # If adding this section would exceed chunk size, finalize current chunk
            if (len(current_chunk) + len(section_content) > self.chunk_size and 
                len(current_chunk) > self.min_chunk_size):
                
                # Create chunk from accumulated content
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        content=current_chunk.strip(),
                        chunk_id=f"{source_file}_chunk_{chunk_index}",
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap from previous
                if chunks and self.overlap > 0:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + "\n\n" + section_content
                    current_start = current_start + len(current_chunk) - len(overlap_text) - len(section_content) - 2
                else:
                    current_chunk = section_content
                    current_start = current_start + len(current_chunk) - len(section_content)
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section_content
                else:
                    current_chunk = section_content
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                chunk_id=f"{source_file}_chunk_{chunk_index}",
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sliding_window_chunk(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Simple sliding window chunking with sentence boundary awareness
        """
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at a sentence boundary if possible
            if end < len(text):
                # Look for sentence endings within last 100 characters
                search_start = max(end - 100, start)
                sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end])]
                
                if sentence_endings:
                    # Use the last sentence ending
                    end = search_start + sentence_endings[-1]
                else:
                    # Look for paragraph breaks
                    para_breaks = [m.start() for m in re.finditer(r'\n\s*\n', text[search_start:end])]
                    if para_breaks:
                        end = search_start + para_breaks[-1]
            
            chunk_content = text[start:end].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    content=chunk_content,
                    chunk_id=f"{source_file}_chunk_{chunk_index}",
                    source_file=source_file,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(end - self.overlap, start + 1)
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        return chunks
    
    def _split_by_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text by structural elements like headers, code blocks, etc.
        """
        sections = []
        current_pos = 0
        
        # Patterns for different structural elements
        patterns = [
            (r'^#{1,6}\s+.+$', 'header'),  # Markdown headers
            (r'```[\w]*\n.*?\n```', 'code_block'),  # Code blocks
            (r'^(\s*[-*+]|\s*\d+\.)\s+.+$', 'list_item'),  # List items
            (r'^\s*$', 'blank_line'),  # Blank lines
        ]
        
        lines = text.split('\n')
        current_section = {'content': '', 'type': 'text'}
        
        for line in lines:
            line_type = 'text'
            
            # Check what type of line this is
            for pattern, ptype in patterns:
                if re.match(pattern, line, re.MULTILINE | re.DOTALL):
                    line_type = ptype
                    break
            
            # If type changed and we have content, save current section
            if line_type != current_section['type'] and current_section['content'].strip():
                sections.append(current_section)
                current_section = {'content': line, 'type': line_type}
            else:
                # Add line to current section
                if current_section['content']:
                    current_section['content'] += '\n' + line
                else:
                    current_section['content'] = line
                current_section['type'] = line_type
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _create_chunk(self, content: str, chunk_id: str, source_file: str,
                     chunk_index: int, start_char: int, end_char: int) -> TextChunk:
        """Create a TextChunk object with metadata"""
        
        # Detect if this is code
        is_code = self._is_code_chunk(content)
        language = self._detect_language(content) if is_code else None
        
        # Create metadata
        metadata = {
            'word_count': len(content.split()),
            'char_count': len(content),
            'has_code': is_code,
            'language': language,
            'structure_type': self._detect_structure_type(content)
        }
        
        return TextChunk(
            content=content,
            chunk_id=chunk_id,
            source_file=source_file,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
            language=language,
            is_code=is_code
        )
    
    def _is_code_chunk(self, content: str) -> bool:
        """Determine if a chunk primarily contains code"""
        code_indicators = [
            r'^```',  # Code block start
            r'def\s+\w+\s*\(',  # Function definition
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statement
            r'#include',  # C/C++ include
            r'function\s+\w+',  # JavaScript function
            r'<\w+[^>]*>.*</\w+>',  # HTML tags
        ]
        
        code_matches = sum(1 for pattern in code_indicators 
                          if re.search(pattern, content, re.MULTILINE | re.IGNORECASE))
        
        # Also check ratio of lines that look like code
        lines = content.split('\n')
        code_lines = sum(1 for line in lines if self._looks_like_code_line(line))
        code_ratio = code_lines / len(lines) if lines else 0
        
        return code_matches > 0 or code_ratio > 0.3
    
    def _looks_like_code_line(self, line: str) -> bool:
        """Check if a single line looks like code"""
        line = line.strip()
        if not line:
            return False
        
        # Common code patterns
        code_patterns = [
            r'^\s*(def|class|import|from|if|for|while|try|except)\s+',
            r'^\s*[\w_]+\s*[=+\-*/]\s*',  # Assignment or operation
            r'^\s*[{}()\[\];]',  # Brackets, braces, semicolons
            r'^\s*#|^\s*//',  # Comments
            r'^\s*<\w+',  # HTML tags
        ]
        
        return any(re.match(pattern, line) for pattern in code_patterns)
    
    def _detect_language(self, content: str) -> Optional[str]:
        """Simple language detection for code chunks"""
        # This is a simplified version - you might want to use the
        # detect_programming_language function from text_processing.py
        if 'def ' in content and 'import ' in content:
            return 'python'
        elif 'function ' in content and ('var ' in content or 'let ' in content):
            return 'javascript'
        elif 'public class ' in content or 'import java.' in content:
            return 'java'
        elif '#include' in content and 'int main' in content:
            return 'cpp'
        elif '<html>' in content or '<div>' in content:
            return 'html'
        elif re.search(r'\w+\s*{\s*[\w\-]+\s*:', content):
            return 'css'
        
        return None
    
    def _detect_structure_type(self, content: str) -> str:
        """Detect the structural type of content"""
        if content.startswith('#'):
            return 'header'
        elif content.startswith('```'):
            return 'code_block'
        elif re.match(r'^\s*[-*+]\s+', content) or re.match(r'^\s*\d+\.\s+', content):
            return 'list'
        elif self._is_code_chunk(content):
            return 'code'
        else:
            return 'text'


def merge_small_chunks(chunks: List[TextChunk], min_size: int = 100) -> List[TextChunk]:
    """
    Merge chunks that are too small with adjacent chunks
    
    Args:
        chunks: List of chunks to process
        min_size: Minimum chunk size in characters
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged_chunks = []
    current_chunk = chunks[0]
    
    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        
        # If current chunk is too small, merge with next
        if len(current_chunk.content) < min_size:
            merged_content = current_chunk.content + "\n\n" + next_chunk.content
            
            # Create new merged chunk
            current_chunk = TextChunk(
                content=merged_content,
                chunk_id=f"{current_chunk.source_file}_merged_{current_chunk.chunk_index}_{next_chunk.chunk_index}",
                source_file=current_chunk.source_file,
                chunk_index=current_chunk.chunk_index,
                start_char=current_chunk.start_char,
                end_char=next_chunk.end_char,
                metadata={
                    **current_chunk.metadata,
                    'merged': True,
                    'original_chunks': [current_chunk.chunk_id, next_chunk.chunk_id]
                },
                language=current_chunk.language or next_chunk.language,
                is_code=current_chunk.is_code or next_chunk.is_code
            )
        else:
            # Current chunk is good size, add to results and move to next
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
    
    # Add the last chunk
    merged_chunks.append(current_chunk)
    
    return merged_chunks