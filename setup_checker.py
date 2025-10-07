# setup_checker.py

import os
from dotenv import load_dotenv
import requests
from typing import Dict, List

class APIKeyChecker:
    """Check and validate API keys for the Programming Helper Agent"""
    
    def __init__(self):
        load_dotenv()
        self.results = {}
    
    def check_all_keys(self) -> Dict:
        """Check all API keys and return status"""
        print("🔑 Programming Helper Agent - API Key Setup Checker")
        print("=" * 60)
        
        # Check each API key
        self.check_groq_api()
        self.check_huggingface_api()
        self.check_judge0_api()
        self.check_github_token()
        self.check_stackoverflow_api()
        self.check_google_translate_api()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def check_groq_api(self):
        """Check GROQ API key (required for AI responses)"""
        groq_key = os.getenv("GROQ_API_KEY")
        
        if groq_key:
            try:
                # Test with a simple request
                from langchain_groq import ChatGroq
                llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")
                # Simple test
                test_response = llm.invoke("Hello")
                
                self.results['groq'] = {
                    'status': 'working',
                    'message': '✅ GROQ API: Working perfectly!',
                    'feature': 'AI-powered Q&A responses',
                    'priority': 'essential'
                }
            except Exception as e:
                self.results['groq'] = {
                    'status': 'error',
                    'message': f'❌ GROQ API: Error - {str(e)[:50]}...',
                    'feature': 'AI-powered Q&A responses',
                    'priority': 'essential'
                }
        else:
            self.results['groq'] = {
                'status': 'missing',
                'message': '⚠️ GROQ API Key: Missing',
                'feature': 'AI-powered Q&A responses',
                'priority': 'essential',
                'setup_url': 'https://console.groq.com/'
            }
    
    def check_huggingface_api(self):
        """Check Hugging Face API token"""
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if hf_token:
            try:
                headers = {"Authorization": f"Bearer {hf_token}"}
                response = requests.get(
                    "https://huggingface.co/api/whoami",
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.results['huggingface'] = {
                        'status': 'working',
                        'message': '✅ Hugging Face API: Working',
                        'feature': 'Enhanced embeddings and models',
                        'priority': 'recommended'
                    }
                else:
                    self.results['huggingface'] = {
                        'status': 'error',
                        'message': f'❌ Hugging Face API: Error {response.status_code}',
                        'feature': 'Enhanced embeddings and models',
                        'priority': 'recommended'
                    }
            except Exception as e:
                self.results['huggingface'] = {
                    'status': 'error',
                    'message': f'❌ Hugging Face API: Connection error',
                    'feature': 'Enhanced embeddings and models',
                    'priority': 'recommended'
                }
        else:
            self.results['huggingface'] = {
                'status': 'missing',
                'message': '⚠️ Hugging Face Token: Missing',
                'feature': 'Enhanced embeddings and models',
                'priority': 'recommended',
                'setup_url': 'https://huggingface.co/settings/tokens'
            }
    
    def check_judge0_api(self):
        """Check Judge0 API key for code execution"""
        judge0_key = os.getenv("JUDGE0_API_KEY")
        
        if judge0_key:
            try:
                headers = {
                    "X-RapidAPI-Key": judge0_key,
                    "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com"
                }
                response = requests.get(
                    "https://judge0-ce.p.rapidapi.com/languages",
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    languages = response.json()
                    self.results['judge0'] = {
                        'status': 'working',
                        'message': f'✅ Judge0 API: Working ({len(languages)} languages available)',
                        'feature': 'Code execution in 70+ languages',
                        'priority': 'high_value'
                    }
                else:
                    self.results['judge0'] = {
                        'status': 'error',
                        'message': f'❌ Judge0 API: Error {response.status_code}',
                        'feature': 'Code execution in 70+ languages',
                        'priority': 'high_value'
                    }
            except Exception as e:
                self.results['judge0'] = {
                    'status': 'error',
                    'message': '❌ Judge0 API: Connection error',
                    'feature': 'Code execution in 70+ languages',
                    'priority': 'high_value'
                }
        else:
            self.results['judge0'] = {
                'status': 'missing',
                'message': '⚠️ Judge0 API Key: Missing',
                'feature': 'Code execution in 70+ languages',
                'priority': 'high_value',
                'setup_url': 'https://rapidapi.com/judge0-official/api/judge0-ce'
            }
    
    def check_github_token(self):
        """Check GitHub personal access token"""
        github_token = os.getenv("GITHUB_TOKEN")
        
        if github_token:
            try:
                headers = {"Authorization": f"token {github_token}"}
                response = requests.get(
                    "https://api.github.com/user",
                    headers=headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    user_data = response.json()
                    self.results['github'] = {
                        'status': 'working',
                        'message': f'✅ GitHub API: Working (User: {user_data.get("login", "Unknown")})',
                        'feature': 'GitHub repository integration',
                        'priority': 'high_value'
                    }
                else:
                    self.results['github'] = {
                        'status': 'error',
                        'message': f'❌ GitHub API: Error {response.status_code}',
                        'feature': 'GitHub repository integration',
                        'priority': 'high_value'
                    }
            except Exception as e:
                self.results['github'] = {
                    'status': 'error',
                    'message': '❌ GitHub API: Connection error',
                    'feature': 'GitHub repository integration',
                    'priority': 'high_value'
                }
        else:
            self.results['github'] = {
                'status': 'missing',
                'message': '⚠️ GitHub Token: Missing',
                'feature': 'GitHub repository integration',
                'priority': 'high_value',
                'setup_url': 'https://github.com/settings/tokens'
            }
    
    def check_stackoverflow_api(self):
        """Check Stack Overflow API key"""
        so_key = os.getenv("STACKOVERFLOW_KEY")
        
        if so_key:
            try:
                response = requests.get(
                    f"https://api.stackexchange.com/2.3/info?site=stackoverflow&key={so_key}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.results['stackoverflow'] = {
                        'status': 'working',
                        'message': '✅ Stack Overflow API: Working',
                        'feature': 'Enhanced programming Q&A search',
                        'priority': 'nice_to_have'
                    }
                else:
                    self.results['stackoverflow'] = {
                        'status': 'error',
                        'message': f'❌ Stack Overflow API: Error {response.status_code}',
                        'feature': 'Enhanced programming Q&A search',
                        'priority': 'nice_to_have'
                    }
            except Exception as e:
                self.results['stackoverflow'] = {
                    'status': 'error',
                    'message': '❌ Stack Overflow API: Connection error',
                    'feature': 'Enhanced programming Q&A search',
                    'priority': 'nice_to_have'
                }
        else:
            self.results['stackoverflow'] = {
                'status': 'missing',
                'message': '⚠️ Stack Overflow Key: Missing',
                'feature': 'Enhanced programming Q&A search',
                'priority': 'nice_to_have',
                'setup_url': 'https://stackapps.com/apps/oauth/register'
            }
    
    def check_google_translate_api(self):
        """Check Google Translate API key"""
        translate_key = os.getenv("GOOGLE_TRANSLATE_KEY")
        
        if translate_key:
            try:
                response = requests.get(
                    f"https://translation.googleapis.com/language/translate/v2/languages?key={translate_key}",
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.results['google_translate'] = {
                        'status': 'working',
                        'message': '✅ Google Translate API: Working',
                        'feature': 'Multi-language documentation support',
                        'priority': 'nice_to_have'
                    }
                else:
                    self.results['google_translate'] = {
                        'status': 'error',
                        'message': f'❌ Google Translate API: Error {response.status_code}',
                        'feature': 'Multi-language documentation support',
                        'priority': 'nice_to_have'
                    }
            except Exception as e:
                self.results['google_translate'] = {
                    'status': 'error',
                    'message': '❌ Google Translate API: Connection error',
                    'feature': 'Multi-language documentation support',
                    'priority': 'nice_to_have'
                }
        else:
            self.results['google_translate'] = {
                'status': 'missing',
                'message': '⚠️ Google Translate Key: Missing',
                'feature': 'Multi-language documentation support',
                'priority': 'nice_to_have',
                'setup_url': 'https://console.cloud.google.com/'
            }
    
    def print_summary(self):
        """Print a comprehensive summary"""
        print("\n" + "=" * 60)
        print("📊 API KEY STATUS SUMMARY")
        print("=" * 60)
        
        # Group by priority
        essential = []
        recommended = []
        nice_to_have = []
        
        for service, info in self.results.items():
            if info['priority'] == 'essential':
                essential.append((service, info))
            elif info['priority'] in ['recommended', 'high_value']:
                recommended.append((service, info))
            else:
                nice_to_have.append((service, info))
        
        # Print essential
        print("\n🚨 ESSENTIAL (Required for core functionality):")
        for service, info in essential:
            print(f"  {info['message']}")
            if info['status'] == 'missing' and 'setup_url' in info:
                print(f"     🔗 Setup: {info['setup_url']}")
        
        # Print recommended
        print("\n⭐ HIGH VALUE (Recommended for best experience):")
        for service, info in recommended:
            print(f"  {info['message']}")
            if info['status'] == 'missing' and 'setup_url' in info:
                print(f"     🔗 Setup: {info['setup_url']}")
        
        # Print nice to have
        print("\n💡 NICE TO HAVE (Optional enhancements):")
        for service, info in nice_to_have:
            print(f"  {info['message']}")
            if info['status'] == 'missing' and 'setup_url' in info:
                print(f"     🔗 Setup: {info['setup_url']}")
        
        # Overall status
        working_count = sum(1 for info in self.results.values() if info['status'] == 'working')
        total_count = len(self.results)
        
        print(f"\n🎯 OVERALL STATUS: {working_count}/{total_count} APIs working")
        
        if working_count == 0:
            print("🔧 START HERE: Add GROQ_API_KEY to get basic AI functionality!")
        elif working_count < 3:
            print("🚀 GOOD START: Add more API keys to unlock additional features!")
        else:
            print("🎉 EXCELLENT: You have great API coverage!")
        
        print("\n📝 SETUP INSTRUCTIONS:")
        print("1. Copy .env.template to .env")
        print("2. Add your API keys to the .env file")
        print("3. Run this checker again to verify")
        print("4. Start your Programming Helper Agent!")
        
        print("\n" + "=" * 60)

def main():
    """Run the API key checker"""
    checker = APIKeyChecker()
    results = checker.check_all_keys()
    
    # Also save results to file for other scripts to use
    import json
    with open('api_key_status.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()