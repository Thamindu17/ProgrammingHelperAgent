# utils/dynamic_document_processor.py

import os
import requests
from typing import List, Dict, Optional
from langchain_community.document_loaders import (
    WebBaseLoader, 
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tempfile
from urllib.parse import urlparse
import time

class DynamicDocumentProcessor:
    """
    Process various types of documentation sources dynamically
    Works without API keys, gets enhanced with them
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Import config safely
        try:
            from config.settings import GITHUB_TOKEN
            self.github_token = GITHUB_TOKEN
        except:
            self.github_token = ""
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Programming-Helper-Agent/1.0'
        })
    
    def detect_source_type(self, source: str) -> str:
        """Detect the type of documentation source"""
        source_lower = source.lower()
        
        if source.startswith('http'):
            if 'github.com' in source:
                if '/blob/' in source or '/raw/' in source:
                    return 'github_file'
                elif '.git' in source or '/tree/' in source:
                    return 'github_repo'
                else:
                    return 'github_file'
            elif source.endswith('.pdf'):
                return 'pdf_url'
            elif any(ext in source for ext in ['.md', '.rst', '.txt']):
                return 'text_url'
            else:
                return 'website'
        elif source.endswith('.pdf'):
            return 'pdf_file'
        elif source.endswith(('.md', '.markdown')):
            return 'markdown_file'
        elif source.endswith('.txt'):
            return 'text_file'
        else:
            return 'unknown'
    
    def process_website(self, url: str) -> List[Document]:
        """Process website/documentation URL using web scraping"""
        try:
            print(f"📄 Processing website: {url}")
            
            # Use requests for better control
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Create document from HTML content
            doc = Document(
                page_content=self._clean_html_content(response.text),
                metadata={
                    'source': url,
                    'type': 'website',
                    'title': self._extract_title(response.text)
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            print(f"✅ Created {len(chunks)} chunks from website")
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing website {url}: {str(e)}")
            return []
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content to extract readable text"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            # Fallback: simple HTML tag removal
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            return title_tag.get_text().strip() if title_tag else "Unknown"
        except:
            return "Unknown"
    
    def process_github_file(self, file_url: str) -> List[Document]:
        """Process single GitHub file (works without API key)"""
        try:
            print(f"🐙 Processing GitHub file: {file_url}")
            
            # Convert to raw URL
            if 'github.com' in file_url and '/blob/' in file_url:
                raw_url = file_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            else:
                raw_url = file_url
            
            response = self.session.get(raw_url, timeout=10)
            response.raise_for_status()
            
            content = response.text
            
            # Extract filename
            filename = file_url.split('/')[-1]
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': file_url,
                    'filename': filename,
                    'type': 'github_file'
                }
            )
            
            chunks = self.text_splitter.split_documents([doc])
            print(f"✅ Created {len(chunks)} chunks from GitHub file")
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing GitHub file {file_url}: {str(e)}")
            return []
    
    def process_github_repo(self, repo_url: str) -> List[Document]:
        """Process GitHub repository (enhanced with API key)"""
        try:
            print(f"🐙 Processing GitHub repository: {repo_url}")
            
            # Extract owner and repo
            parts = repo_url.replace('https://github.com/', '').replace('.git', '').split('/')
            if len(parts) < 2:
                print("❌ Invalid GitHub repository URL")
                return []
            
            owner, repo = parts[0], parts[1]
            
            if self.github_token:
                return self._process_github_repo_with_api(owner, repo, repo_url)
            else:
                return self._process_github_repo_without_api(owner, repo, repo_url)
                
        except Exception as e:
            print(f"❌ Error processing GitHub repo {repo_url}: {str(e)}")
            return []
    
    def _process_github_repo_with_api(self, owner: str, repo: str, repo_url: str) -> List[Document]:
        """Process GitHub repo using API (when token available)"""
        try:
            headers = {'Authorization': f'token {self.github_token}'}
            
            # Get repository contents
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
            response = self.session.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                documents = []
                files = response.json()
                
                # Process documentation files
                doc_files = [f for f in files if self._is_documentation_file(f['name'])]
                
                for file_info in doc_files[:10]:  # Limit to 10 files
                    if file_info['type'] == 'file':
                        file_content = self._download_github_file(file_info['download_url'], headers)
                        if file_content:
                            doc = Document(
                                page_content=file_content,
                                metadata={
                                    'source': f"{repo_url}/{file_info['name']}",
                                    'filename': file_info['name'],
                                    'repo': f"{owner}/{repo}",
                                    'method': 'github_api'
                                }
                            )
                            documents.append(doc)
                
                chunks = self.text_splitter.split_documents(documents)
                print(f"✅ GitHub API: Created {len(chunks)} chunks from {len(documents)} files")
                return chunks
            else:
                print(f"❌ GitHub API error: {response.status_code}")
                return self._process_github_repo_without_api(owner, repo, repo_url)
                
        except Exception as e:
            print(f"❌ GitHub API error: {e}")
            return self._process_github_repo_without_api(owner, repo, repo_url)
    
    def _process_github_repo_without_api(self, owner: str, repo: str, repo_url: str) -> List[Document]:
        """Process GitHub repo without API (fallback method)"""
        try:
            print("📄 Using fallback method (no GitHub API key)")
            
            # Try to get README file
            readme_urls = [
                f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
                f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
                f"https://raw.githubusercontent.com/{owner}/{repo}/main/readme.md",
                f"https://raw.githubusercontent.com/{owner}/{repo}/master/readme.md"
            ]
            
            documents = []
            
            for readme_url in readme_urls:
                try:
                    response = self.session.get(readme_url, timeout=5)
                    if response.status_code == 200:
                        doc = Document(
                            page_content=response.text,
                            metadata={
                                'source': readme_url,
                                'filename': 'README.md',
                                'repo': f"{owner}/{repo}",
                                'method': 'direct_download'
                            }
                        )
                        documents.append(doc)
                        print(f"✅ Found README at: {readme_url}")
                        break
                except:
                    continue
            
            if documents:
                chunks = self.text_splitter.split_documents(documents)
                print(f"✅ Fallback method: Created {len(chunks)} chunks")
                return chunks
            else:
                print("❌ Could not find README file")
                return []
                
        except Exception as e:
            print(f"❌ Fallback method error: {e}")
            return []
    
    def _is_documentation_file(self, filename: str) -> bool:
        """Check if file is likely documentation"""
        doc_keywords = ['readme', 'doc', 'guide', 'tutorial', 'example', 'getting', 'started']
        doc_extensions = ['.md', '.rst', '.txt']
        
        filename_lower = filename.lower()
        return (any(keyword in filename_lower for keyword in doc_keywords) or 
                any(filename_lower.endswith(ext) for ext in doc_extensions))
    
    def _download_github_file(self, download_url: str, headers: dict = None) -> Optional[str]:
        """Download file content from GitHub"""
        try:
            response = self.session.get(download_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error downloading file: {e}")
        return None
    
    def process_text_url(self, url: str) -> List[Document]:
        """Process text file from URL"""
        try:
            print(f"📄 Processing text file: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            doc = Document(
                page_content=response.text,
                metadata={
                    'source': url,
                    'type': 'text_url',
                    'filename': url.split('/')[-1]
                }
            )
            
            chunks = self.text_splitter.split_documents([doc])
            print(f"✅ Created {len(chunks)} chunks from text file")
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing text file {url}: {str(e)}")
            return []
    
    def process_uploaded_file(self, uploaded_file, file_content: str) -> List[Document]:
        """Process uploaded file content"""
        try:
            print(f"📁 Processing uploaded file: {uploaded_file.name}")
            
            doc = Document(
                page_content=file_content,
                metadata={
                    'source': uploaded_file.name,
                    'type': 'uploaded_file',
                    'filename': uploaded_file.name,
                    'size': len(file_content)
                }
            )
            
            chunks = self.text_splitter.split_documents([doc])
            print(f"✅ Created {len(chunks)} chunks from uploaded file")
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing uploaded file: {str(e)}")
            return []
    
    def process_documentation(self, source: str, source_type: str = None) -> List[Document]:
        """Main method to process any documentation source"""
        if source_type is None:
            source_type = self.detect_source_type(source)
        
        print(f"🔍 Processing {source_type}: {source}")
        
        # Add small delay to be respectful
        time.sleep(0.5)
        
        if source_type == 'website':
            return self.process_website(source)
        elif source_type == 'github_repo':
            return self.process_github_repo(source)
        elif source_type == 'github_file':
            return self.process_github_file(source)
        elif source_type == 'text_url':
            return self.process_text_url(source)
        else:
            print(f"❌ Unsupported source type: {source_type}")
            return []

# Test the processor
if __name__ == "__main__":
    processor = DynamicDocumentProcessor()
    
    # Test with a simple URL
    test_url = "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md"
    docs = processor.process_documentation(test_url)
    print(f"Test result: {len(docs)} documents processed")