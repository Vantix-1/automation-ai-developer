"""
AI Chat Engine - Core intelligence handling conversation flow and memory
"""
import openai
from typing import List, Dict, Optional
from .memory_manager import ConversationMemory

class AIChatEngine:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.memory = ConversationMemory()
        
    def chat(self, message: str, user_id: str = "default") -> str:
        """Process user message and return AI response"""
        try:
            # Get conversation history
            history = self.memory.get_conversation(user_id)
            
            # Build messages with system prompt and history
            messages = self._build_messages(history, message)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
            # Update conversation memory
            self.memory.add_message(user_id, "user", message)
            self.memory.add_message(user_id, "assistant", ai_response)
            
            return ai_response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _build_messages(self, history: List[Dict], new_message: str) -> List[Dict]:
        """Build message list for API call"""
        system_prompt = {
            "role": "system", 
            "content": "You are a helpful AI assistant. Provide clear, concise responses and maintain context from previous messages."
        }
        
        messages = [system_prompt]
        
        # Add conversation history (last 10 exchanges)
        for msg in history[-20:]:  # Keep last 10 exchanges
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add new user message
        messages.append({"role": "user", "content": new_message})
        
        return messages