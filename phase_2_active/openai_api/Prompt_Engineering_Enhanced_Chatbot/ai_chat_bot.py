"""
Enhanced AI Chatbot - Day 3-5: Prompt Engineering & Advanced Features
"""

import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Message:
    role: str
    content: str
    timestamp: str
    tokens: int = 0
    
    def to_dict(self):
        return asdict(self)

class PromptTemplate:
    TEMPLATES = {
        "coding_assistant": {
            "system": """You are an expert Python coding assistant. Help with:
1. Writing clean, efficient Python code
2. Debugging and explaining errors
3. Code optimization and best practices
4. Algorithm design and implementation

Always provide:
- Code with proper syntax highlighting
- Clear explanations
- Alternative approaches when relevant
- Time and space complexity analysis""",
            "user_example": "How do I implement binary search in Python?",
            "temperature": 0.3
        },
        "creative_writer": {
            "system": """You are a creative writer with expertise in multiple genres.
Be imaginative, descriptive, and maintain consistent characterization.""",
            "user_example": "Write a short story about a robot learning to paint",
            "temperature": 0.8
        },
        "learning_coach": {
            "system": """You are a patient, encouraging learning coach.
Focus on building understanding rather than just providing answers.""",
            "user_example": "Explain how neural networks work to a beginner",
            "temperature": 0.5
        }
    }
    
    @classmethod
    def get_template(cls, name: str):
        return cls.TEMPLATES.get(name)
    
    @classmethod
    def list_templates(cls):
        return list(cls.TEMPLATES.keys())

class EnhancedAIChatBot:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)
        self.conversation: List[Message] = []
        self.context_window = 16385
        self.stats = {
            "total_tokens": 0,
            "total_messages": 0,
            "total_cost": 0.0,
            "start_time": datetime.datetime.now().isoformat(),
            "templates_used": {}
        }
        self.current_template = "coding_assistant"
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def add_message(self, role: str, content: str):
        tokens = self.count_tokens(content)
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.datetime.now().isoformat(),
            tokens=tokens
        )
        self.conversation.append(message)
        self.stats["total_tokens"] += tokens
        self.stats["total_messages"] += 1
        return message
    
    def apply_prompt_template(self, template_name: str):
        template = PromptTemplate.get_template(template_name)
        if not template:
            print(f"Template '{template_name}' not found")
            return False
        
        self.conversation = []
        self.add_message("system", template["system"])
        self.current_template = template_name
        self.stats["templates_used"][template_name] = \
            self.stats["templates_used"].get(template_name, 0) + 1
        
        print(f"✅ Applied template: {template_name}")
        return True
    
    def generate_response(self, user_input: str, stream: bool = True) -> str:
        self.add_message("user", user_input)
        
        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation
        ]
        
        template = PromptTemplate.get_template(self.current_template)
        temperature = template["temperature"] if template else 0.7
        
        try:
            if stream:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=1000,
                    stream=True
                )
                
                collected_message = ""
                print("🤖 Assistant: ", end="", flush=True)
                
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        print(chunk_content, end="", flush=True)
                        collected_message += chunk_content
                
                print()
                self.add_message("assistant", collected_message)
                return collected_message
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=1000
                )
                
                assistant_message = response.choices[0].message.content
                print(f"🤖 Assistant: {assistant_message}")
                self.add_message("assistant", assistant_message)
                return assistant_message
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""

def interactive_chat_demo():
    print("\n" + "="*70)
    print("🚀 Enhanced AI Chatbot - Day 3-5 Demo")
    print("="*70)
    
    try:
        chatbot = EnhancedAIChatBot()
        
        print("\n📚 Available Templates:")
        templates = PromptTemplate.list_templates()
        for i, template in enumerate(templates, 1):
            print(f"  {i}. {template}")
        
        choice = input("\n🎯 Select template (number or name): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(templates):
            template_name = templates[int(choice) - 1]
            chatbot.apply_prompt_template(template_name)
        elif choice in templates:
            chatbot.apply_prompt_template(choice)
        
        print("\n💬 Chat Started (type 'quit' to exit)")
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            chatbot.generate_response(user_input)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_chat_demo()