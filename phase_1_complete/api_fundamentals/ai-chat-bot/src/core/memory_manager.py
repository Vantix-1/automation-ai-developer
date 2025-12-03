"""
Conversation Memory Manager - Handles context and chat history
"""
from typing import List, Dict, Optional
from datetime import datetime
import json
import os

class ConversationMemory:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict]] = {}
        
    def add_message(self, user_id: str, role: str, content: str):
        """Add a message to conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversations[user_id].append(message)
        
        # Trim history if too long
        if len(self.conversations[user_id]) > self.max_history * 2:  # *2 for user/assistant pairs
            self.conversations[user_id] = self.conversations[user_id][-self.max_history * 2:]
    
    def get_conversation(self, user_id: str) -> List[Dict]:
        """Get conversation history for user"""
        return self.conversations.get(user_id, [])
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for user"""
        if user_id in self.conversations:
            self.conversations[user_id] = []
    
    def save_conversation(self, user_id: str, filepath: str):
        """Save conversation to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.conversations.get(user_id, []), f, indent=2)