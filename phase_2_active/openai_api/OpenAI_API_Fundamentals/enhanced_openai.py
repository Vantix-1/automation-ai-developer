"""
OpenAI API Mastery - Day 1-2 Enhanced
Advanced features: Cost calculator, JSON export, and Streamlit web interface
"""

import os
import sys
import json
import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Add parent directory to path for shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class OpenAIConnector:
    """Enhanced OpenAI connector with cost tracking and export capabilities"""
    
    # Pricing per 1K tokens (as of Jan 2025)
    PRICING = {
        "gpt-3.5-turbo": {
            "input": 0.0015,    # $0.0015 per 1K tokens input
            "output": 0.0020,   # $0.0020 per 1K tokens output
        },
        "gpt-3.5-turbo-16k": {
            "input": 0.0030,
            "output": 0.0040,
        },
        "gpt-4": {
            "input": 0.0300,
            "output": 0.0600,
        },
        "gpt-4-turbo-preview": {
            "input": 0.0100,
            "output": 0.0300,
        },
        "gpt-4-32k": {
            "input": 0.0600,
            "output": 0.1200,
        }
    }
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client with API key"""
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "start_time": datetime.datetime.now().isoformat(),
            "model": model
        }
        
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = None) -> float:
        """Calculate cost based on token usage"""
        model = model or self.model
        if model not in self.PRICING:
            print(f"⚠️  Pricing not available for {model}, using gpt-3.5-turbo pricing")
            model = "gpt-3.5-turbo"
            
        pricing = self.PRICING[model]
        prompt_cost = (prompt_tokens / 1000) * pricing["input"]
        completion_cost = (completion_tokens / 1000) * pricing["output"]
        
        return round(prompt_cost + completion_cost, 6)
    
    def update_stats(self, usage: Dict[str, Any]):
        """Update session statistics"""
        self.session_stats["total_tokens"] += usage.total_tokens
        self.session_stats["prompt_tokens"] += usage.prompt_tokens
        self.session_stats["completion_tokens"] += usage.completion_tokens
        
        cost = self.calculate_cost(
            usage.prompt_tokens, 
            usage.completion_tokens,
            usage.model if hasattr(usage, 'model') else self.model
        )
        self.session_stats["total_cost"] += cost
        
    def test_connection(self):
        """Test basic API connection with cost tracking"""
        try:
            print("🔌 Testing OpenAI API connection...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello and confirm you're connected."}
                ],
                max_tokens=30,
                temperature=0.5
            )
            
            # Update statistics
            self.update_stats(response.usage)
            
            print(f"✅ Connection successful!")
            print(f"   Model: {response.model}")
            print(f"   Response: {response.choices[0].message.content}")
            print(f"   Tokens used: {response.usage.total_tokens}")
            print(f"   Cost: ${self.calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens, response.model):.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def interactive_chat(self, system_prompt: str = "You are a helpful AI assistant."):
        """Enhanced interactive chat with cost tracking"""
        print("\n" + "="*60)
        print("🤖 OpenAI Interactive Chat with Cost Tracking")
        print("="*60)
        print("Commands:")
        print("  'quit' - Exit chat")
        print("  'clear' - Clear conversation")
        print("  'stats' - Show session statistics")
        print("  'export' - Export conversation to JSON")
        print("  'system <prompt>' - Change system prompt")
        print("  'model <model_name>' - Switch model")
        print("-"*60)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    self.show_session_stats()
                    break
                    
                elif user_input.lower() == 'clear':
                    messages = [{"role": "system", "content": system_prompt}]
                    self.conversation_history = []
                    print("🧹 Conversation cleared")
                    continue
                    
                elif user_input.lower() == 'stats':
                    self.show_session_stats()
                    continue
                    
                elif user_input.lower() == 'export':
                    self.export_to_json()
                    continue
                    
                elif user_input.startswith('system '):
                    system_prompt = user_input[7:]
                    messages = [{"role": "system", "content": system_prompt}]
                    print(f"⚙️ System prompt updated: {system_prompt[:50]}...")
                    continue
                    
                elif user_input.startswith('model '):
                    new_model = user_input[6:].strip()
                    if new_model in self.PRICING:
                        self.model = new_model
                        print(f"🔄 Model switched to: {new_model}")
                    else:
                        print(f"❌ Model '{new_model}' not recognized or pricing not available")
                        print(f"   Available models: {', '.join(self.PRICING.keys())}")
                    continue
                
                # Add user message to conversation
                messages.append({"role": "user", "content": user_input})
                
                # Get AI response
                print("⏳ Thinking...", end='', flush=True)
                
                response = self.client.chat.completions.create(
                    model=self.model,
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
                
                # Update statistics (we need to estimate tokens since streaming doesn't provide usage)
                # In a real implementation, you'd want to use tiktoken for accurate counting
                estimated_prompt_tokens = len(user_input) // 4  # Rough estimate
                estimated_completion_tokens = len(full_response) // 4
                cost = self.calculate_cost(estimated_prompt_tokens, estimated_completion_tokens)
                self.session_stats["total_cost"] += cost
                self.session_stats["total_tokens"] += estimated_prompt_tokens + estimated_completion_tokens
                
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": full_response})
                
                # Store in conversation history
                exchange = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model,
                    "user_message": user_input,
                    "assistant_message": full_response,
                    "estimated_tokens": {
                        "prompt": estimated_prompt_tokens,
                        "completion": estimated_completion_tokens,
                        "total": estimated_prompt_tokens + estimated_completion_tokens
                    },
                    "estimated_cost": cost
                }
                self.conversation_history.append(exchange)
                
                print(f"📊 Estimated cost: ${cost:.6f}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_session_stats(self):
        """Display detailed session statistics"""
        print("\n" + "="*60)
        print("📊 Session Statistics")
        print("="*60)
        print(f"Model: {self.session_stats['model']}")
        print(f"Start Time: {self.session_stats['start_time']}")
        print(f"Total Tokens: {self.session_stats['total_tokens']:,}")
        print(f"Total Cost: ${self.session_stats['total_cost']:.6f}")
        print(f"Exchanges: {len(self.conversation_history)}")
        
        if self.conversation_history:
            avg_tokens = self.session_stats['total_tokens'] / len(self.conversation_history)
            avg_cost = self.session_stats['total_cost'] / len(self.conversation_history)
            print(f"Avg per exchange: {avg_tokens:.1f} tokens, ${avg_cost:.6f}")
        
        # Show cost breakdown
        print("\n💰 Cost Breakdown by Model (estimated):")
        model_costs = {}
        for exchange in self.conversation_history:
            model = exchange['model']
            cost = exchange.get('estimated_cost', 0)
            model_costs[model] = model_costs.get(model, 0) + cost
        
        for model, cost in model_costs.items():
            percentage = (cost / self.session_stats['total_cost'] * 100) if self.session_stats['total_cost'] > 0 else 0
            print(f"  {model}: ${cost:.6f} ({percentage:.1f}%)")
        
        print("="*60)
    
    def export_to_json(self, filename: Optional[str] = None):
        """Export conversation history to JSON file"""
        if not self.conversation_history:
            print("❌ No conversation history to export")
            return
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_export_{timestamp}.json"
        
        export_data = {
            "metadata": {
                "export_date": datetime.datetime.now().isoformat(),
                "model": self.session_stats["model"],
                "total_exchanges": len(self.conversation_history),
                "total_tokens": self.session_stats["total_tokens"],
                "total_cost": self.session_stats["total_cost"],
                "session_start": self.session_stats["start_time"]
            },
            "pricing_reference": self.PRICING,
            "conversation": self.conversation_history
        }
        
        try:
            # Create exports directory if it doesn't exist
            exports_dir = Path("exports")
            exports_dir.mkdir(exist_ok=True)
            
            filepath = exports_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Conversation exported to: {filepath}")
            print(f"   Total exchanges: {len(self.conversation_history)}")
            print(f"   File size: {filepath.stat().st_size:,} bytes")
            
            # Also create a simplified text version
            self.export_to_text(filepath.with_suffix('.txt'))
            
        except Exception as e:
            print(f"❌ Error exporting to JSON: {e}")
    
    def export_to_text(self, filepath: Path):
        """Export conversation to readable text format"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("OpenAI Conversation Export\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.session_stats['model']}\n")
                f.write(f"Total Exchanges: {len(self.conversation_history)}\n")
                f.write(f"Total Tokens: {self.session_stats['total_tokens']:,}\n")
                f.write(f"Total Cost: ${self.session_stats['total_cost']:.6f}\n\n")
                
                for i, exchange in enumerate(self.conversation_history, 1):
                    f.write(f"\n{'='*40}\n")
                    f.write(f"Exchange #{i}\n")
                    f.write(f"{'='*40}\n")
                    f.write(f"Timestamp: {exchange['timestamp']}\n")
                    f.write(f"Model: {exchange['model']}\n")
                    
                    if 'estimated_tokens' in exchange:
                        tokens = exchange['estimated_tokens']
                        f.write(f"Tokens: {tokens['prompt']} (prompt) + {tokens['completion']} (completion) = {tokens['total']} (total)\n")
                    
                    if 'estimated_cost' in exchange:
                        f.write(f"Cost: ${exchange['estimated_cost']:.6f}\n")
                    
                    f.write(f"\n👤 User:\n{exchange['user_message']}\n\n")
                    f.write(f"🤖 Assistant:\n{exchange['assistant_message']}\n")
                
                f.write("\n" + "="*70 + "\n")
                f.write("End of Conversation\n")
                f.write("="*70 + "\n")
            
            print(f"📝 Text version created: {filepath}")
            
        except Exception as e:
            print(f"⚠️ Error creating text export: {e}")
    
    def explore_models_with_pricing(self):
        """Display available models with detailed pricing"""
        print("\n" + "="*70)
        print("📚 Available OpenAI Models with Pricing")
        print("="*70)
        
        for model, pricing in self.PRICING.items():
            print(f"\n🔹 {model}")
            print(f"   💰 Pricing per 1K tokens:")
            print(f"      Input:  ${pricing['input']:.4f}")
            print(f"      Output: ${pricing['output']:.4f}")
            print(f"      Example 100-token chat: ${(pricing['input'] + pricing['output']) / 10:.6f}")
        
        print("\n" + "="*70)
        print("💡 Cost Estimation Examples:")
        print("="*70)
        
        examples = [
            ("Short chat (~50 tokens)", 50, "gpt-3.5-turbo"),
            ("Medium article (~1000 tokens)", 1000, "gpt-3.5-turbo"),
            ("Long analysis (~5000 tokens)", 5000, "gpt-4"),
            ("Document processing (~10000 tokens)", 10000, "gpt-4-turbo-preview"),
        ]
        
        for desc, tokens, model in examples:
            if model in self.PRICING:
                pricing = self.PRICING[model]
                cost = (tokens / 1000) * (pricing["input"] + pricing["output"])
                print(f"{desc}: ${cost:.4f} with {model}")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🚀 OpenAI API Mastery - Enhanced Day 1-2")
    print("="*70)
    
    try:
        # Initialize connector
        connector = OpenAIConnector()
        
        # Test connection
        if not connector.test_connection():
            return
        
        # Explore models with pricing
        connector.explore_models_with_pricing()
        
        # Start interactive chat
        start_chat = input("\n🎯 Start interactive chat with cost tracking? (y/n): ").lower()
        if start_chat == 'y':
            system_prompt = input("Enter system prompt (or press Enter for default): ")
            if not system_prompt:
                system_prompt = "You are a helpful AI assistant with cost awareness."
            
            connector.interactive_chat(system_prompt)
            
            # Offer to export conversation
            if connector.conversation_history:
                export = input("\n💾 Export conversation to JSON? (y/n): ").lower()
                if export == 'y':
                    connector.export_to_json()
        
        print(f"\n✅ Enhanced Day 1-2 complete!")
        connector.show_session_stats()
        
        # Check if we should run Streamlit web interface
        run_streamlit = input("\n🌐 Run Streamlit web interface? (y/n): ").lower()
        if run_streamlit == 'y':
            print("Starting Streamlit app...")
            print("Please run: streamlit run streamlit_app.py")
            print("(Make sure Streamlit is installed: pip install streamlit)")
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("Please ensure you have:")
        print("1. Created a .env file with OPENAI_API_KEY")
        print("2. Installed required packages")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()