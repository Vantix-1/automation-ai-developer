"""
🧠 Memory Systems Implementation
Day 15: Conversation memory and state management
"""

import json
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain.chains import ConversationChain

load_dotenv()

class MemoryManager:
    """Production conversation memory management"""
    
    def __init__(self, memory_type: str = "buffer"):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory_type = memory_type
        self.conversation_history = []
        self._setup_memory()
    
    def _setup_memory(self):
        """Configure memory based on type"""
        if self.memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        elif self.memory_type == "window":
            self.memory = ConversationBufferWindowMemory(
                k=3,  # Keep last 3 exchanges
                return_messages=True,
                memory_key="chat_history"
            )
        elif self.memory_type == "summary":
            self.memory = ConversationSummaryMemory.from_messages(
                llm=self.llm,
                memory_key="chat_history"
            )
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")
        
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def chat(self, message: str) -> str:
        """Process message with memory"""
        # Track conversation
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": message
        })
        
        # Get response
        response = self.chain.predict(input=message)
        
        # Track response
        self.conversation_history[-1]["ai"] = response
        
        return response
    
    def get_context(self) -> str:
        """Get current conversation context"""
        memory_vars = self.memory.load_memory_variables({})
        return str(memory_vars.get("chat_history", ""))
    
    def save_conversation(self, filename: str = "conversation.json"):
        """Save conversation to file"""
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        print(f"💾 Saved conversation to {filename}")
    
    def clear_memory(self):
        """Clear all memory"""
        self.memory.clear()
        self.conversation_history = []
        print("🧹 Memory cleared")

class AdvancedMemorySystem:
    """Memory with file persistence and summary"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        
        # Multiple memory systems
        self.short_term = ConversationBufferWindowMemory(k=5)
        self.long_term = ConversationSummaryMemory(llm=self.llm)
        
        self.history_file = "memory_history.json"
        self._load_history()
    
    def _load_history(self):
        """Load conversation history from file"""
        try:
            with open(self.history_file, 'r') as f:
                self.conversation_history = json.load(f)
            print(f"📖 Loaded {len(self.conversation_history)} previous conversations")
        except FileNotFoundError:
            self.conversation_history = []
    
    def _save_history(self):
        """Save conversation history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def process_message(self, message: str) -> str:
        """Process message with advanced memory"""
        
        # Get context from both memories
        short_context = self.short_term.load_memory_variables({})
        long_context = self.long_term.load_memory_variables({})
        
        # Enhanced prompt with memory
        prompt = f"""
        Previous conversation summary: {long_context.get('history', 'No summary yet')}
        
        Recent conversation: {short_context.get('history', 'No recent conversation')}
        
        New message: {message}
        
        Respond appropriately considering the conversation history.
        """
        
        # Get response
        response = self.llm.predict(prompt)
        
        # Update both memories
        self.short_term.save_context(
            {"input": message},
            {"output": response}
        )
        
        self.long_term.save_context(
            {"input": message},
            {"output": response}
        )
        
        # Save to history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": message,
            "ai": response
        })
        self._save_history()
        
        return response
    
    def get_conversation_summary(self) -> str:
        """Get summary of entire conversation"""
        if not self.conversation_history:
            return "No conversation history"
        
        # Use LLM to create summary
        summary_prompt = f"""
        Summarize this conversation:
        
        {json.dumps(self.conversation_history[-10:], indent=2)}
        
        Provide a concise summary of key topics discussed.
        """
        
        return self.llm.predict(summary_prompt)

def demo_memory_systems():
    """Demonstrate memory systems"""
    print("=" * 50)
    print("🧠 MEMORY SYSTEMS DEMO (Day 15)")
    print("=" * 50)
    
    print("\n1️⃣ Buffer Memory (Remembers everything):")
    buffer_chat = MemoryManager(memory_type="buffer")
    
    conversations = [
        "Hi, I'm Alex. I love hiking and photography.",
        "What's your favorite hiking trail?",
        "Tell me about your photography equipment."
    ]
    
    for msg in conversations:
        response = buffer_chat.chat(msg)
        print(f"You: {msg}")
        print(f"AI: {response[:80]}...")
    
    print(f"\nBuffer Context: {buffer_chat.get_context()[:200]}...")
    buffer_chat.save_conversation("buffer_conversation.json")
    
    print("\n2️⃣ Window Memory (Last 3 exchanges):")
    window_chat = MemoryManager(memory_type="window")
    
    for msg in conversations:
        window_chat.chat(msg)
    
    print(f"Window Context: {window_chat.get_context()[:200]}...")
    
    print("\n3️⃣ Advanced Memory with Summary:")
    advanced = AdvancedMemorySystem()
    
    response = advanced.process_message("What outdoor activities do you recommend?")
    print(f"Response: {response}")
    
    print(f"\nConversation Summary:")
    print(advanced.get_conversation_summary())

if __name__ == "__main__":
    demo_memory_systems()