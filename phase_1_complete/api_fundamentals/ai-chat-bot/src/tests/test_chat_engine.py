"""
Test suite for AI Chat Engine
"""
import pytest
from src.core.chat_engine import AIChatEngine
from src.core.memory_manager import ConversationMemory

class TestChatEngine:
    def test_memory_management(self):
        """Test conversation memory functionality"""
        memory = ConversationMemory()
        memory.add_message("user1", "user", "Hello")
        memory.add_message("user1", "assistant", "Hi there!")
        
        history = memory.get_conversation("user1")
        assert len(history) == 2
        assert history[0]["content"] == "Hello"
    
    def test_message_building(self, mock_openai):
        """Test message structure for API calls"""
        chat_engine = AIChatEngine("test-key")
        messages = chat_engine._build_messages([], "Test message")
        
        assert len(messages) == 2  # system prompt + user message
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Test message"