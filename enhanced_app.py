# enhanced_app.py

import gradio as gr
from agent.enhanced_agent import EnhancedProgrammingAgent
import json

print("🚀 Initializing Enhanced Programming Helper Agent...")
print("Loading enhanced capabilities with free APIs...")

# Initialize the enhanced agent
agent = EnhancedProgrammingAgent()

# Check capabilities status
capabilities = agent.get_capabilities_status()
print(f"✅ Enhanced agent ready with capabilities: {capabilities['enabled_features']}")

if capabilities['disabled_features']:
    print(f"⚠️  Disabled features: {capabilities['disabled_features']}")
    print("💡 Recommendations:")
    for rec in capabilities['recommendations']:
        print(f"   - {rec}")

def process_enhanced_query(question, language="auto-detect", enable_code_execution=True):
    """
    Enhanced query processing with multiple sources
    """
    print(f"🔍 Processing query: {question}")
    
    # Prepare context
    context = {
        'language': language if language != "auto-detect" else None,
        'enable_execution': enable_code_execution
    }
    
    # Process with enhanced agent
    result = agent.process_query(question, context)
    
    # Format response for display
    formatted_response = format_enhanced_response(result)
    
    return formatted_response

def format_enhanced_response(result):
    """Format the enhanced response for better display"""
    
    # Main answer
    response = f"## 🤖 Main Answer\n\n{result['main_answer']}\n\n"
    
    # Programming language detected
    if result.get('programming_language'):
        response += f"**🔍 Detected Language:** {result['programming_language'].title()}\n\n"
    
    # Sources used
    sources = result.get('sources', {})
    sources_text = []
    if sources.get('knowledge_base'):
        sources_text.append("📚 Knowledge Base")
    if sources.get('stackoverflow'):
        sources_text.append("🔗 Stack Overflow")
    if sources.get('github'):
        sources_text.append("🐙 GitHub")
    if sources.get('code_execution'):
        sources_text.append("▶️ Code Execution")
    
    if sources_text:
        response += f"**📊 Sources Used:** {' | '.join(sources_text)}\n\n"
    
    # Code execution results
    if result.get('code_execution'):
        exec_result = result['code_execution']
        if exec_result.get('output'):
            response += f"## ▶️ Code Execution Output\n\n```\n{exec_result['output']}\n```\n\n"
        if exec_result.get('error'):
            response += f"## ❌ Execution Error\n\n```\n{exec_result['error']}\n```\n\n"
    
    # Stack Overflow results
    so_data = result.get('additional_resources', {}).get('stackoverflow')
    if so_data and so_data.get('top_question'):
        question = so_data['top_question']
        response += f"## 🔗 Related Stack Overflow Question\n\n"
        response += f"**[{question['title']}]({question['link']})**\n\n"
        response += f"Score: {question['score']} | Answers: {question['answer_count']}\n\n"
        if question.get('body'):
            response += f"{question['body'][:300]}...\n\n"
    
    # GitHub results
    gh_data = result.get('additional_resources', {}).get('github')
    if gh_data and gh_data.get('code_examples'):
        response += f"## 🐙 GitHub Code Examples\n\n"
        for i, example in enumerate(gh_data['code_examples'][:2], 1):
            response += f"**{i}. [{example['repository']}]({example['repository_url']})**\n"
            response += f"File: `{example['file_name']}` | ⭐ {example['stars']} stars\n\n"
            if example.get('code_snippet'):
                response += f"```{example.get('language', '').lower()}\n{example['code_snippet'][:200]}...\n```\n\n"
    
    # API Status (for debugging)
    api_status = result.get('api_status', {})
    if any('error' in str(status) for status in api_status.values()):
        response += f"## ⚠️ API Status\n\n"
        for api, status in api_status.items():
            if 'error' in str(status):
                response += f"- {api}: {status}\n"
        response += "\n"
    
    return response

def get_code_execution_result(code, language="python"):
    """Standalone code execution function"""
    result = agent.code_executor.execute_code(code, language)
    
    if result.get('output'):
        return f"✅ Output:\n{result['output']}"
    elif result.get('error'):
        return f"❌ Error:\n{result['error']}"
    else:
        return f"⚠️ Status: {result.get('status', 'Unknown')}"

def show_capabilities():
    """Show current capabilities and setup instructions"""
    status = agent.get_capabilities_status()
    
    response = "## 🔧 Current Capabilities\n\n"
    
    # Enabled features
    if status['enabled_features']:
        response += "### ✅ Enabled Features\n"
        for feature in status['enabled_features']:
            response += f"- {feature.replace('_', ' ').title()}\n"
        response += "\n"
    
    # Disabled features
    if status['disabled_features']:
        response += "### ❌ Disabled Features\n"
        for feature in status['disabled_features']:
            response += f"- {feature.replace('_', ' ').title()}\n"
        response += "\n"
    
    # Setup recommendations
    if status['recommendations']:
        response += "### 💡 Setup Recommendations (Free APIs)\n"
        for rec in status['recommendations']:
            response += f"- {rec}\n"
        response += "\n"
    
    # API setup instructions
    response += """### 🔑 Free API Setup Instructions

**1. Judge0 (Code Execution) - Free: 50 requests/day**
- Go to: https://rapidapi.com/judge0-official/api/judge0-ce
- Sign up for free account
- Add `JUDGE0_API_KEY=your_key` to .env file

**2. GitHub (Code Search) - Free: 5,000 requests/hour**
- Go to: https://github.com/settings/tokens
- Create personal access token
- Add `GITHUB_TOKEN=your_token` to .env file

**3. Stack Overflow (Enhanced Search) - Free: 10,000 requests/day**
- Go to: https://stackapps.com/apps/oauth/register
- Register your app
- Add `STACKOVERFLOW_KEY=your_key` to .env file

All these APIs offer generous free tiers!"""
    
    return response

# Create the enhanced Gradio interface
with gr.Blocks(title="🤖 Enhanced Programming Helper Agent", theme=gr.themes.Soft()) as enhanced_iface:
    gr.Markdown("# 🤖 Enhanced Programming Helper Agent")
    gr.Markdown("*Powered by multiple free APIs for comprehensive programming assistance*")
    
    with gr.Tab("💬 Ask Questions"):
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    lines=3,
                    placeholder="e.g., How do I merge two DataFrames in Pandas?",
                    label="Your Programming Question"
                )
                with gr.Row():
                    language_dropdown = gr.Dropdown(
                        choices=["auto-detect", "python", "javascript", "java", "cpp", "c", "csharp", "go", "rust", "php"],
                        value="auto-detect",
                        label="Programming Language"
                    )
                    code_exec_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Code Execution"
                    )
                submit_btn = gr.Button("🔍 Get Enhanced Answer", variant="primary")
            
            with gr.Column(scale=1):
                capabilities_btn = gr.Button("🔧 Show Capabilities", variant="secondary")
        
        answer_output = gr.Markdown(label="Enhanced Response")
        
        submit_btn.click(
            process_enhanced_query,
            inputs=[question_input, language_dropdown, code_exec_checkbox],
            outputs=answer_output
        )
        
        capabilities_btn.click(
            show_capabilities,
            outputs=answer_output
        )
    
    with gr.Tab("▶️ Code Execution"):
        gr.Markdown("## Execute Code Directly")
        
        with gr.Row():
            with gr.Column():
                code_input = gr.Code(
                    language="python",
                    placeholder="# Enter your code here\nprint('Hello, World!')",
                    label="Code to Execute"
                )
                exec_language = gr.Dropdown(
                    choices=["python", "javascript", "java", "cpp", "c"],
                    value="python",
                    label="Language"
                )
                execute_btn = gr.Button("▶️ Run Code", variant="primary")
            
            with gr.Column():
                execution_output = gr.Textbox(
                    lines=10,
                    label="Execution Result",
                    interactive=False
                )
        
        execute_btn.click(
            get_code_execution_result,
            inputs=[code_input, exec_language],
            outputs=execution_output
        )
    
    with gr.Tab("📊 Features"):
        gr.Markdown(show_capabilities())

# Launch the enhanced interface
if __name__ == "__main__":
    enhanced_iface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )