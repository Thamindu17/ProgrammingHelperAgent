"""
File processing utilities for the Programming Helper Agent
Handles reading and extracting text from various file formats
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Import statements for different file types (will need to be installed)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class FileProcessor:
    """Utility class for processing different file types"""
    
    def __init__(self, supported_extensions: Optional[List[str]] = None):
        """
        Initialize file processor
        
        Args:
            supported_extensions: List of file extensions to process
        """
        self.supported_extensions = supported_extensions or [
            '.txt', '.md', '.py', '.js', '.html', '.css', '.json', 
            '.pdf', '.docx', '.rst', '.java', '.cpp', '.c', '.h'
        ]
        
        # Track processed files to avoid duplicates
        self.processed_hashes = set()
    
    def get_file_list(self, directory: Union[str, Path], 
                     recursive: bool = True) -> List[Path]:
        """
        Get list of supported files in directory
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of Path objects for supported files
        """
        directory = Path(directory)
        if not directory.exists():
            return []
        
        files = []
        
        if recursive:
            # Recursively find all files
            for ext in self.supported_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            # Only current directory
            for ext in self.supported_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        # Filter out hidden files and directories
        files = [f for f in files if not any(part.startswith('.') for part in f.parts)]
        
        return sorted(files)
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Process a single file and extract text content
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with file metadata and content, or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist")
            return None
        
        if file_path.suffix.lower() not in self.supported_extensions:
            print(f"Warning: File extension {file_path.suffix} not supported")
            return None
        
        try:
            # Get file metadata
            stat = file_path.stat()
            file_hash = self._get_file_hash(file_path)
            
            # Check if already processed
            if file_hash in self.processed_hashes:
                print(f"Skipping already processed file: {file_path}")
                return None
            
            # Extract content based on file type
            content = self._extract_content(file_path)
            
            if content is None:
                return None
            
            # Mark as processed
            self.processed_hashes.add(file_hash)
            
            # Create file metadata
            file_data = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_extension': file_path.suffix.lower(),
                'file_size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'file_hash': file_hash,
                'content': content,
                'content_length': len(content),
                'processed_at': datetime.now().isoformat()
            }
            
            return file_data
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def _extract_content(self, file_path: Path) -> Optional[str]:
        """Extract text content from file based on its type"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._extract_pdf_content(file_path)
            elif extension == '.docx':
                return self._extract_docx_content(file_path)
            elif extension in ['.txt', '.md', '.py', '.js', '.html', '.css', 
                             '.json', '.rst', '.java', '.cpp', '.c', '.h']:
                return self._extract_text_content(file_path)
            else:
                print(f"No extraction method for {extension}")
                return None
                
        except Exception as e:
            print(f"Error extracting content from {file_path}: {str(e)}")
            return None
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from plain text files"""
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading {file_path} with {encoding}: {str(e)}")
                continue
        
        # If all encodings fail, try binary mode and decode with errors='ignore'
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Failed to read {file_path} in binary mode: {str(e)}")
            return ""
    
    def _extract_pdf_content(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF files"""
        if not PDF_AVAILABLE:
            print("PyPDF2 not available. Install with: pip install PyPDF2")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(f"=== Page {page_num + 1} ===\n{text}")
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1} from {file_path}: {str(e)}")
                        continue
                
                return "\n\n".join(text_content)
                
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return None
    
    def _extract_docx_content(self, file_path: Path) -> Optional[str]:
        """Extract text from DOCX files"""
        if not DOCX_AVAILABLE:
            print("python-docx not available. Install with: pip install python-docx")
            return None
        
        try:
            doc = Document(file_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {str(e)}")
            return None
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash of file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error hashing file {file_path}: {str(e)}")
            return str(file_path)  # Fallback to file path
    
    def batch_process_directory(self, directory: Union[str, Path], 
                               recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory
        
        Args:
            directory: Directory to process
            recursive: Whether to process subdirectories
            
        Returns:
            List of processed file data
        """
        files = self.get_file_list(directory, recursive)
        processed_files = []
        
        print(f"Found {len(files)} files to process...")
        
        for i, file_path in enumerate(files, 1):
            print(f"Processing {i}/{len(files)}: {file_path.name}")
            
            file_data = self.process_file(file_path)
            if file_data:
                processed_files.append(file_data)
        
        print(f"Successfully processed {len(processed_files)} files")
        return processed_files
    
    def save_processed_data(self, processed_files: List[Dict[str, Any]], 
                           output_file: Union[str, Path]):
        """
        Save processed file data to JSON file
        
        Args:
            processed_files: List of processed file data
            output_file: Path to save the data
        """
        output_file = Path(output_file)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_files, f, indent=2, ensure_ascii=False)
            
            print(f"Saved processed data to {output_file}")
            
        except Exception as e:
            print(f"Error saving processed data: {str(e)}")
    
    def load_processed_data(self, input_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load previously processed file data from JSON
        
        Args:
            input_file: Path to the JSON file
            
        Returns:
            List of processed file data
        """
        input_file = Path(input_file)
        
        if not input_file.exists():
            print(f"File {input_file} does not exist")
            return []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update processed hashes to avoid reprocessing
            for file_data in data:
                if 'file_hash' in file_data:
                    self.processed_hashes.add(file_data['file_hash'])
            
            print(f"Loaded {len(data)} processed files from {input_file}")
            return data
            
        except Exception as e:
            print(f"Error loading processed data: {str(e)}")
            return []


def get_file_stats(file_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about processed files
    
    Args:
        file_list: List of processed file data
        
    Returns:
        Dictionary with file statistics
    """
    if not file_list:
        return {}
    
    total_files = len(file_list)
    total_size = sum(f.get('file_size', 0) for f in file_list)
    total_content_length = sum(f.get('content_length', 0) for f in file_list)
    
    # Count by extension
    extensions = {}
    for f in file_list:
        ext = f.get('file_extension', 'unknown')
        extensions[ext] = extensions.get(ext, 0) + 1
    
    # Average content length
    avg_content_length = total_content_length / total_files if total_files > 0 else 0
    
    return {
        'total_files': total_files,
        'total_size_bytes': total_size,
        'total_content_length': total_content_length,
        'average_content_length': avg_content_length,
        'extensions': extensions,
        'largest_file': max(file_list, key=lambda x: x.get('content_length', 0)) if file_list else None,
        'smallest_file': min(file_list, key=lambda x: x.get('content_length', 0)) if file_list else None
    }