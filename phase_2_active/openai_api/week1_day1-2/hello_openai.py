"""
OpenAI API Mastery - Day 1-2
First API Connection and Basic Chat Implementation
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path for shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class OpenAIConnector:
    """Handles OpenAI API connection and basic operations"""
    
    def __init__(self):
        """Initialize OpenAI client with API key"""
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.total_tokens_used = 0
        
    def test_connection(self, model="gpt-3.5-turbo"):
        """Test basic API connection"""
        try:
            print("🔌 Testing OpenAI API connection...")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello and confirm you're connected."}
                ],
                max_tokens=30,
                temperature=0.5
            )
            
            print(f"✅ Connection successful!")
            print(f"   Model: {response.model}")
            print(f"   Response: {response.choices[0].message.content}")
            print(f"   Tokens: {response.usage.total_tokens}")
            
            self.total_tokens_used += response.usage.total_tokens
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def simple_chat(self, system_prompt="You are a helpful assistant."):
        """Interactive chat interface"""
        print("\n" + "="*50)
        print("🤖 OpenAI Interactive Chat")
        print("="*50)
        print("Commands:")
        print("  'quit' - Exit chat")
        print("  'clear' - Clear conversation history")
        print("  'tokens' - Show token usage")
        print("  'system <prompt>' - Change system prompt")
        print("-"*50)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'clear':
                    messages = [{"role": "system", "content": system_prompt}]
                    self.conversation_history = []
                    print("🧹 Conversation cleared")
                    continue
                    
                elif user_input.lower() == 'tokens':
                    print(f"📊 Total tokens used: {self.total_tokens_used}")
                    continue
                    
                elif user_input.startswith('system '):
                    system_prompt = user_input[7:]
                    messages = [{"role": "system", "content": system_prompt}]
                    print(f"⚙️ System prompt updated: {system_prompt[:50]}...")
                    continue
                
                # Add user message to conversation
                messages.append({"role": "user", "content": user_input})
                
                # Get AI response
                print("⏳ Thinking...", end='', flush=True)
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                    stream=True
                )
                
                # Stream the response
                print("\r🤖 AI: ", end='', flush=True)
                full_response = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        full_response += content
                
                print()  # New line after streaming
                
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": full_response})
                
                # Store in conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": full_response,
                    "timestamp": os.path.getmtime(__file__)
                })
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def explore_models(self):
        """Display information about available models"""
        models_info = {
            "gpt-3.5-turbo": {
                "description": "Most capable GPT-3.5 model",
                "max_tokens": 16385,
                "training_data": "Up to Sep 2021",
                "best_for": ["Chat", "Content generation", "Code completion"],
                "cost_per_1k_tokens": 0.0015  # Input tokens
            },
            "gpt-3.5-turbo-16k": {
                "description": "Same capabilities but 16K context",
                "max_tokens": 16385,
                "training_data": "Up to Sep 2021",
                "best_for": ["Long conversations", "Documents"],
                "cost_per_1k_tokens": 0.003
            },
            "gpt-4": {
                "description": "More capable than any GPT-3.5 model",
                "max_tokens": 8192,
                "training_data": "Up to Sep 2021",
                "best_for": ["Complex reasoning", "Advanced coding", "Creative tasks"],
                "cost_per_1k_tokens": 0.03
            },
            "gpt-4-turbo-preview": {
                "description": "Latest GPT-4 model with 128K context",
                "max_tokens": 128000,
                "training_data": "Up to Apr 2023",
                "best_for": ["Very long context", "Complex analysis"],
                "cost_per_1k_tokens": 0.01
            }
        }
        
        print("\n" + "="*60)
        print("📚 Available OpenAI Models")
        print("="*60)
        
        for model, info in models_info.items():
            print(f"\n🔹 {model}")
            print(f"   📝 {info['description']}")
            print(f"   📏 Context: {info['max_tokens']:,} tokens")
            print(f"   📅 Training data: {info['training_data']}")
            print(f"   💰 Cost per 1K tokens: ${info['cost_per_1k_tokens']}")
            print(f"   🎯 Best for: {', '.join(info['best_for'])}")
        
        print("\n💡 Tip: Start with gpt-3.5-turbo for testing to save costs")
    
    def save_conversation(self, filename="conversation_history.txt"):
        """Save conversation history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("OpenAI Conversation History\n")
                f.write("="*60 + "\n\n")
                
                for i, exchange in enumerate(self.conversation_history, 1):
                    f.write(f"Exchange {i}:\n")
                    f.write(f"  👤 User: {exchange['user']}\n")
                    f.write(f"  🤖 AI: {exchange['assistant']}\n")
                    f.write("-"*40 + "\n")
                
                f.write(f"\n📊 Total tokens used: {self.total_tokens_used}\n")
                f.write("="*60 + "\n")
            
            print(f"💾 Conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🚀 OpenAI API Mastery - Day 1-2")
    print("="*60)
    
    try:
        # Initialize connector
        connector = OpenAIConnector()
        
        # Test connection
        if not connector.test_connection():
            return
        
        # Explore models
        connector.explore_models()
        
        # Start interactive chat
        start_chat = input("\n🎯 Start interactive chat? (y/n): ").lower()
        if start_chat == 'y':
            system_prompt = input("Enter system prompt (or press Enter for default): ")
            if not system_prompt:
                system_prompt = "You are a helpful AI assistant."
            
            connector.simple_chat(system_prompt)
            
            # Option to save conversation
            save = input("\n💾 Save conversation history? (y/n): ").lower()
            if save == 'y':
                connector.save_conversation()
        
        print(f"\n✅ Day 1-2 complete!")
        print(f"📊 Total tokens used in session: {connector.total_tokens_used}")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Please ensure you have:")
        print("1. Created a .env file with OPENAI_API_KEY")
        print("2. Installed required packages: pip install openai python-dotenv")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()