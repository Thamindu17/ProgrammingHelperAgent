# services/github_search.py

import requests
from typing import List, Dict, Optional
from config.settings import GITHUB_TOKEN, GITHUB_API, ENABLE_GITHUB_SEARCH

class GitHubSearcher:
    """
    Free GitHub API integration for code examples
    Free tier: 5,000 requests per hour (authenticated)
    """
    
    def __init__(self):
        self.token = GITHUB_TOKEN
        self.base_url = GITHUB_API
        self.enabled = ENABLE_GITHUB_SEARCH
        self.headers = {}
        
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
            self.headers['Accept'] = 'application/vnd.github.v3+json'
    
    def search_code(self, query: str, language: str = None, limit: int = 5) -> List[Dict]:
        """
        Search for code examples on GitHub
        
        Args:
            query: Search query (e.g., "pandas merge dataframe")
            language: Programming language filter
            limit: Maximum number of results
            
        Returns:
            List of code example dictionaries
        """
        if not self.enabled and not self.token:
            return [{
                "error": "GitHub search disabled. Add GITHUB_TOKEN to .env for enhanced code examples",
                "code": "",
                "repository": "",
                "file_path": ""
            }]
        
        try:
            search_query = query
            if language:
                search_query += f" language:{language}"
            
            params = {
                'q': search_query,
                'sort': 'indexed',
                'order': 'desc',
                'per_page': limit
            }
            
            response = requests.get(
                f"{self.base_url}/search/code",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._format_code_results(data.get('items', []))
            elif response.status_code == 403:
                return [{
                    "error": "GitHub API rate limit exceeded. Please try again later.",
                    "code": "",
                    "repository": "",
                    "file_path": ""
                }]
            else:
                return []
                
        except Exception as e:
            print(f"GitHub search error: {e}")
            return []
    
    def get_file_content(self, repo_full_name: str, file_path: str, ref: str = "main") -> Optional[str]:
        """Get the full content of a file from GitHub"""
        try:
            url = f"{self.base_url}/repos/{repo_full_name}/contents/{file_path}"
            params = {'ref': ref}
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                import base64
                content = response.json().get('content', '')
                return base64.b64decode(content).decode('utf-8')
            
            return None
            
        except Exception as e:
            print(f"GitHub file content error: {e}")
            return None
    
    def _format_code_results(self, items: List[Dict]) -> List[Dict]:
        """Format GitHub search results"""
        formatted = []
        for item in items:
            repo = item.get('repository', {})
            formatted.append({
                'repository': repo.get('full_name', ''),
                'file_path': item.get('path', ''),
                'file_name': item.get('name', ''),
                'code_snippet': item.get('text_matches', [{}])[0].get('fragment', '') if item.get('text_matches') else '',
                'html_url': item.get('html_url', ''),
                'repository_url': repo.get('html_url', ''),
                'stars': repo.get('stargazers_count', 0),
                'language': repo.get('language', ''),
                'description': repo.get('description', ''),
                'score': item.get('score', 0)
            })
        return formatted
    
    def search_repositories(self, query: str, language: str = None, limit: int = 5) -> List[Dict]:
        """Search for repositories related to the query"""
        try:
            search_query = query
            if language:
                search_query += f" language:{language}"
            
            params = {
                'q': search_query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': limit
            }
            
            response = requests.get(
                f"{self.base_url}/search/repositories",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._format_repo_results(data.get('items', []))
            else:
                return []
                
        except Exception as e:
            print(f"GitHub repository search error: {e}")
            return []
    
    def _format_repo_results(self, items: List[Dict]) -> List[Dict]:
        """Format repository search results"""
        formatted = []
        for item in items:
            formatted.append({
                'name': item.get('name', ''),
                'full_name': item.get('full_name', ''),
                'description': item.get('description', ''),
                'html_url': item.get('html_url', ''),
                'stars': item.get('stargazers_count', 0),
                'forks': item.get('forks_count', 0),
                'language': item.get('language', ''),
                'topics': item.get('topics', []),
                'updated_at': item.get('updated_at', ''),
                'score': item.get('score', 0)
            })
        return formatted
    
    def get_popular_examples(self, programming_language: str, topic: str) -> Dict:
        """
        Get popular code examples for a specific topic and language
        
        Args:
            programming_language: Language to search for
            topic: Topic/keyword to search for
            
        Returns:
            Combined results with code examples and repositories
        """
        code_results = self.search_code(topic, programming_language, limit=3)
        repo_results = self.search_repositories(topic, programming_language, limit=3)
        
        return {
            "code_examples": code_results,
            "related_repositories": repo_results,
            "search_context": {
                "language": programming_language,
                "topic": topic
            }
        }

# Example usage
if __name__ == "__main__":
    searcher = GitHubSearcher()
    results = searcher.get_popular_examples("python", "pandas merge")
    print("GitHub Results:", results)