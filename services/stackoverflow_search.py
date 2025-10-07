# services/stackoverflow_search.py

import requests
from typing import List, Dict, Optional
from config.settings import STACKOVERFLOW_KEY, STACKOVERFLOW_API, ENABLE_STACKOVERFLOW_SEARCH

class StackOverflowSearcher:
    """
    Free Stack Overflow API integration
    Free tier: 10,000 requests per day
    """
    
    def __init__(self):
        self.api_key = STACKOVERFLOW_KEY
        self.base_url = STACKOVERFLOW_API
        self.enabled = ENABLE_STACKOVERFLOW_SEARCH
    
    def search_questions(self, query: str, tags: List[str] = None, limit: int = 5) -> List[Dict]:
        """
        Search Stack Overflow questions
        
        Args:
            query: Search query
            tags: List of tags to filter by (e.g., ['python', 'pandas'])
            limit: Maximum number of results
            
        Returns:
            List of question dictionaries
        """
        if not self.enabled and not self.api_key:
            # Even without API key, we can use the public API with lower limits
            pass
        
        try:
            params = {
                'order': 'desc',
                'sort': 'relevance',
                'intitle': query,
                'site': 'stackoverflow',
                'pagesize': limit,
                'filter': 'withbody'  # Include question body
            }
            
            if tags:
                params['tagged'] = ';'.join(tags)
            
            if self.api_key:
                params['key'] = self.api_key
            
            response = requests.get(f"{self.base_url}/questions", params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_questions(data.get('items', []))
            else:
                return []
                
        except Exception as e:
            print(f"Stack Overflow search error: {e}")
            return []
    
    def get_answers(self, question_id: int) -> List[Dict]:
        """Get answers for a specific question"""
        try:
            params = {
                'order': 'desc',
                'sort': 'votes',
                'site': 'stackoverflow',
                'filter': 'withbody'
            }
            
            if self.api_key:
                params['key'] = self.api_key
            
            response = requests.get(
                f"{self.base_url}/questions/{question_id}/answers",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._format_answers(data.get('items', []))
            else:
                return []
                
        except Exception as e:
            print(f"Stack Overflow answers error: {e}")
            return []
    
    def _format_questions(self, questions: List[Dict]) -> List[Dict]:
        """Format question data for display"""
        formatted = []
        for q in questions:
            formatted.append({
                'id': q.get('question_id'),
                'title': q.get('title', ''),
                'body': q.get('body', '')[:500] + '...' if len(q.get('body', '')) > 500 else q.get('body', ''),
                'score': q.get('score', 0),
                'answer_count': q.get('answer_count', 0),
                'tags': q.get('tags', []),
                'link': q.get('link', ''),
                'is_answered': q.get('is_answered', False)
            })
        return formatted
    
    def _format_answers(self, answers: List[Dict]) -> List[Dict]:
        """Format answer data for display"""
        formatted = []
        for a in answers:
            formatted.append({
                'id': a.get('answer_id'),
                'body': a.get('body', ''),
                'score': a.get('score', 0),
                'is_accepted': a.get('is_accepted', False)
            })
        return formatted
    
    def search_with_context(self, query: str, programming_language: str = None) -> Dict:
        """
        Enhanced search with programming language context
        
        Args:
            query: Search query
            programming_language: Programming language to focus on
            
        Returns:
            Formatted search results with context
        """
        tags = []
        if programming_language:
            tags.append(programming_language.lower())
        
        questions = self.search_questions(query, tags=tags, limit=3)
        
        if not questions:
            return {
                "found": False,
                "message": "No relevant Stack Overflow questions found",
                "questions": []
            }
        
        # Get answers for the top question
        top_question = questions[0]
        if top_question['is_answered']:
            answers = self.get_answers(top_question['id'])
            top_question['answers'] = answers[:2]  # Top 2 answers
        
        return {
            "found": True,
            "total_questions": len(questions),
            "questions": questions,
            "search_context": {
                "query": query,
                "language": programming_language,
                "tags": tags
            }
        }

# Example usage
if __name__ == "__main__":
    searcher = StackOverflowSearcher()
    results = searcher.search_with_context("merge dataframes", "python")
    print("Search Results:", results)