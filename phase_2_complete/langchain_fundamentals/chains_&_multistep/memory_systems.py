"""
🧠 Memory Systems Implementation (Modern LCEL)
Day 15: Conversation memory for AI assistants
"""

import os
import json
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

class MemorySystemsDemo:
    """Demonstrate different memory systems"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        self.parser = StrOutputParser()
    
    def buffer_memory_demo(self):
        """Complete conversation history (Buffer Memory)"""
        print("\n" + "="*50)
        print("1️⃣ BUFFER MEMORY (Complete History)")
        print("="*50)
        
        # Create chat history
        history = ChatMessageHistory()
        
        # Prompt with history
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use conversation history to provide context-aware responses."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Chain
        chain = prompt | self.llm | self.parser
        
        # Conversation
        conversations = [
            "My name is Alice and I love Python programming",
            "What's my name?",
            "What programming language do I like?"
        ]
        
        for user_msg in conversations:
            print(f"\n👤 User: {user_msg}")
            
            # Invoke with history
            response = chain.invoke({
                "history": history.messages,
                "input": user_msg
            })
            
            print(f"🤖 Assistant: {response}")
            
            # Update history
            history.add_user_message(user_msg)
            history.add_ai_message(response)
        
        print(f"\n📊 Total messages in memory: {len(history.messages)}")
        return history
    
    def window_memory_demo(self, window_size: int = 4):
        """Sliding window memory (last N messages)"""
        print("\n" + "="*50)
        print(f"2️⃣ WINDOW MEMORY (Last {window_size//2} exchanges)")
        print("="*50)
        
        history = ChatMessageHistory()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm | self.parser
        
        conversations = [
            "I live in New York",
            "I work as a data scientist",
            "My favorite color is blue",
            "Where do I live?",  # Should remember
            "What's my favorite color?"  # Might not remember if window is too small
        ]
        
        for user_msg in conversations:
            print(f"\n👤 User: {user_msg}")
            
            # Get only last N messages (window)
            window_messages = history.messages[-window_size:] if len(history.messages) > window_size else history.messages
            
            response = chain.invoke({
                "history": window_messages,
                "input": user_msg
            })
            
            print(f"🤖 Assistant: {response}")
            print(f"   📊 Window size: {len(window_messages)} messages")
            
            history.add_user_message(user_msg)
            history.add_ai_message(response)
        
        return history
    
    def summary_memory_demo(self):
        """Summarized conversation memory"""
        print("\n" + "="*50)
        print("3️⃣ SUMMARY MEMORY (Summarized History)")
        print("="*50)
        
        history = ChatMessageHistory()
        conversation_summary = ""
        
        # Summary prompt
        summary_prompt = ChatPromptTemplate.from_template(
            """Summarize this conversation concisely:

{conversation}

Keep only the most important information."""
        )
        
        summary_chain = summary_prompt | self.llm | self.parser
        
        # Chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Here's a summary of our previous conversation: {summary}"),
            ("human", "{input}")
        ])
        
        chat_chain = chat_prompt | self.llm | self.parser
        
        conversations = [
            "I'm planning a trip to Japan in March",
            "I'm interested in visiting Tokyo and Kyoto",
            "I love trying local food and visiting temples",
            "What should I pack for my trip?"
        ]
        
        for i, user_msg in enumerate(conversations):
            print(f"\n👤 User: {user_msg}")
            
            # Generate response
            response = chat_chain.invoke({
                "summary": conversation_summary if conversation_summary else "No previous conversation",
                "input": user_msg
            })
            
            print(f"🤖 Assistant: {response}")
            
            # Add to history
            history.add_user_message(user_msg)
            history.add_ai_message(response)
            
            # Create summary every 2 exchanges
            if (i + 1) % 2 == 0:
                conversation_text = "\n".join([
                    f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
                    for msg in history.messages
                ])
                
                conversation_summary = summary_chain.invoke({
                    "conversation": conversation_text
                })
                
                print(f"\n📝 Updated Summary: {conversation_summary[:100]}...")
        
        return conversation_summary
    
    def file_persistence_demo(self):
        """Save and load conversation history from file"""
        print("\n" + "="*50)
        print("4️⃣ FILE PERSISTENCE (Save/Load History)")
        print("="*50)
        
        history_file = "conversation_history.json"
        
        # Create new conversation
        history = ChatMessageHistory()
        history.add_user_message("My favorite movie is Inception")
        history.add_ai_message("Inception is a great film! What do you like about it?")
        history.add_user_message("I love the complex plot and visual effects")
        history.add_ai_message("The dream layers and cinematography are indeed masterful!")
        
        # Save to file
        messages_data = [
            {
                "type": msg.type,
                "content": msg.content
            }
            for msg in history.messages
        ]
        
        with open(history_file, 'w') as f:
            json.dump(messages_data, f, indent=2)
        
        print(f"💾 Saved {len(history.messages)} messages to {history_file}")
        
        # Load from file
        with open(history_file, 'r') as f:
            loaded_data = json.load(f)
        
        new_history = ChatMessageHistory()
        for msg_data in loaded_data:
            if msg_data["type"] == "human":
                new_history.add_user_message(msg_data["content"])
            else:
                new_history.add_ai_message(msg_data["content"])
        
        print(f"📂 Loaded {len(new_history.messages)} messages from file")
        print("\n📜 Conversation History:")
        for msg in new_history.messages:
            role = "👤 User" if msg.type == "human" else "🤖 Assistant"
            print(f"{role}: {msg.content}")
        
        # Clean up
        if os.path.exists(history_file):
            os.remove(history_file)
            print(f"\n🗑️ Cleaned up {history_file}")
        
        return new_history

def run_all_demos():
    """Run all memory system demonstrations"""
    print("=" * 50)
    print("🧠 MEMORY SYSTEMS DEMO (Day 15)")
    print("=" * 50)
    
    demo = MemorySystemsDemo()
    
    # Run each demo
    demo.buffer_memory_demo()
    demo.window_memory_demo(window_size=4)
    demo.summary_memory_demo()
    demo.file_persistence_demo()
    
    print("\n" + "="*50)
    print("✅ All memory demos complete!")
    print("="*50)
    print("\n📚 Key Takeaways:")
    print("  • Buffer Memory: Stores complete history")
    print("  • Window Memory: Keeps only recent messages")
    print("  • Summary Memory: Compresses history into summary")
    print("  • File Persistence: Save/load conversations")

if __name__ == "__main__":
    run_all_demos()