# services/code_executor.py

import requests
import time
import base64
from typing import Dict, Optional
from config.settings import JUDGE0_API_KEY, JUDGE0_ENDPOINT, ENABLE_CODE_EXECUTION

class CodeExecutor:
    """
    Free code execution service using Judge0 API
    Free tier: 50 submissions per day
    """
    
    def __init__(self):
        self.api_key = JUDGE0_API_KEY
        self.endpoint = JUDGE0_ENDPOINT
        self.enabled = ENABLE_CODE_EXECUTION
        
        # Language IDs for Judge0 API
        self.language_ids = {
            'python': 71,
            'javascript': 63,
            'java': 62,
            'cpp': 54,
            'c': 50,
            'csharp': 51,
            'go': 60,
            'rust': 73,
            'php': 68,
            'ruby': 72,
            'kotlin': 78,
            'swift': 83,
            'typescript': 74
        }
    
    def execute_code(self, code: str, language: str, stdin: str = "") -> Dict:
        """
        Execute code using Judge0 API
        
        Args:
            code: Source code to execute
            language: Programming language
            stdin: Standard input for the program
            
        Returns:
            Dict with execution results
        """
        if not self.enabled:
            return {
                "error": "Code execution disabled. Please add JUDGE0_API_KEY to .env file",
                "output": "",
                "status": "disabled"
            }
        
        if language.lower() not in self.language_ids:
            return {
                "error": f"Language '{language}' not supported",
                "output": "",
                "status": "unsupported"
            }
        
        try:
            # Prepare submission
            submission_data = {
                "source_code": base64.b64encode(code.encode()).decode(),
                "language_id": self.language_ids[language.lower()],
                "stdin": base64.b64encode(stdin.encode()).decode() if stdin else ""
            }
            
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
                "Content-Type": "application/json"
            }
            
            # Submit code for execution
            response = requests.post(
                f"{self.endpoint}/submissions",
                json=submission_data,
                headers=headers,
                params={"base64_encoded": "true", "wait": "false"}
            )
            
            if response.status_code != 201:
                return {
                    "error": f"Submission failed: {response.text}",
                    "output": "",
                    "status": "failed"
                }
            
            submission_token = response.json()["token"]
            
            # Poll for results
            return self._poll_results(submission_token, headers)
            
        except Exception as e:
            return {
                "error": f"Execution error: {str(e)}",
                "output": "",
                "status": "error"
            }
    
    def _poll_results(self, token: str, headers: Dict, max_attempts: int = 10) -> Dict:
        """Poll for execution results"""
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"{self.endpoint}/submissions/{token}",
                    headers=headers,
                    params={"base64_encoded": "true"}
                )
                
                if response.status_code != 200:
                    time.sleep(1)
                    continue
                
                result = response.json()
                status_id = result.get("status", {}).get("id")
                
                # Status IDs: 1-2 = in queue/processing, 3 = accepted, 4+ = various errors
                if status_id in [1, 2]:
                    time.sleep(1)
                    continue
                
                # Decode base64 results
                output = base64.b64decode(result.get("stdout", "") or "").decode()
                error = base64.b64decode(result.get("stderr", "") or "").decode()
                compile_output = base64.b64decode(result.get("compile_output", "") or "").decode()
                
                return {
                    "output": output,
                    "error": error or compile_output,
                    "status": result.get("status", {}).get("description", "Unknown"),
                    "time": result.get("time"),
                    "memory": result.get("memory"),
                    "status_id": status_id
                }
                
            except Exception as e:
                time.sleep(1)
                continue
        
        return {
            "error": "Execution timeout - please try again",
            "output": "",
            "status": "timeout"
        }

# Example usage
if __name__ == "__main__":
    executor = CodeExecutor()
    
    # Test Python code
    python_code = """
print("Hello, World!")
for i in range(3):
    print(f"Count: {i}")
"""
    
    result = executor.execute_code(python_code, "python")
    print("Python Result:", result)