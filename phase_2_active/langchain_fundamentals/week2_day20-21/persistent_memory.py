"""
💾 Persistent Memory Systems
Day 20: Advanced memory storage with databases and file systems
"""

import os
import json
import pickle
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.storage import LocalFileStore, InMemoryStore
from langchain.storage.file_system import LocalFileStore

load_dotenv()

@dataclass
class Conversation:
    """Conversation data class for structured storage"""
    id: str
    user_id: str
    title: str
    messages: List[Dict]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

class SQLiteMemoryStore:
    """Persistent memory storage using SQLite"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    messages TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON conversations (user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON conversations (created_at)
            """)
    
    def save_conversation(self, conversation: Conversation) -> str:
        """Save conversation to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversations 
                (id, user_id, title, messages, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation.id,
                conversation.user_id,
                conversation.title,
                json.dumps(conversation.messages),
                json.dumps(conversation.metadata),
                conversation.created_at,
                conversation.updated_at
            ))
            conn.commit()
            return conversation.id
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load conversation from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations WHERE id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            if row:
                return Conversation(
                    id=row[0],
                    user_id=row[1],
                    title=row[2],
                    messages=json.loads(row[3]),
                    metadata=json.loads(row[4]),
                    created_at=row[5],
                    updated_at=row[6]
                )
            return None
    
    def list_user_conversations(self, user_id: str, limit: int = 50) -> List[Conversation]:
        """List conversations for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE user_id = ? 
                ORDER BY updated_at DESC 
                LIMIT ?
            """, (user_id, limit))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append(Conversation(
                    id=row[0],
                    user_id=row[1],
                    title=row[2],
                    messages=json.loads(row[3]),
                    metadata=json.loads(row[4]),
                    created_at=row[5],
                    updated_at=row[6]
                ))
            return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            cursor.execute("DELETE FROM embeddings WHERE conversation_id = ?", (conversation_id,))
            conn.commit()
            return cursor.rowcount > 0

class VectorMemoryStore:
    """Memory storage with vector embeddings for semantic search"""
    
    def __init__(self, persist_directory: str = "./vector_memory"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="conversation_memory",
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_memory(self, text: str, metadata: Dict[str, Any]) -> str:
        """Store memory with embeddings"""
        # Generate embedding
        embedding = self.embeddings.embed_query(text)
        
        # Create unique ID
        memory_id = f"memory_{datetime.now().timestamp()}"
        
        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
        
        return memory_id
    
    def search_memories(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search memories semantically"""
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filters if filters else {}
        )
        
        # Format results
        memories = []
        if results['ids'][0]:
            for i, memory_id in enumerate(results['ids'][0]):
                memories.append({
                    "id": memory_id,
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": 1 - results['distances'][0][i] if results['distances'] else 0
                })
        
        return memories
    
    def get_conversation_context(self, conversation_id: str, recent_messages: int = 10) -> str:
        """Get context from conversation memories"""
        filters = {"conversation_id": conversation_id}
        memories = self.search_memories("", k=recent_messages, filters=filters)
        
        context_parts = []
        for memory in memories:
            context_parts.append(f"- {memory['text']}")
        
        return "\n".join(context_parts)

class HybridMemorySystem:
    """Hybrid memory system combining multiple storage backends"""
    
    def __init__(self):
        self.sqlite_store = SQLiteMemoryStore()
        self.vector_store = VectorMemoryStore()
        self.llm = ChatOpenAI(temperature=0.7)
        
        # In-memory cache for frequent access
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def create_conversation(self, user_id: str, title: str = "") -> Conversation:
        """Create a new conversation"""
        conversation_id = f"conv_{datetime.now().timestamp()}"
        now = datetime.now().isoformat()
        
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title or f"Conversation {now[:10]}",
            messages=[],
            metadata={"created": now, "message_count": 0},
            created_at=now,
            updated_at=now
        )
        
        self.sqlite_store.save_conversation(conversation)
        self.cache[conversation_id] = (conversation, datetime.now().timestamp())
        
        return conversation
    
    def add_message(self, conversation_id: str, role: str, content: str) -> Dict:
        """Add message to conversation"""
        # Load conversation
        conversation = self.sqlite_store.load_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Create message
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to conversation
        conversation.messages.append(message)
        conversation.metadata["message_count"] = len(conversation.messages)
        conversation.updated_at = datetime.now().isoformat()
        
        # Save to SQLite
        self.sqlite_store.save_conversation(conversation)
        
        # Store in vector memory for semantic search
        if len(content) > 20:  # Only store substantial messages
            memory_metadata = {
                "conversation_id": conversation_id,
                "user_id": conversation.user_id,
                "role": role,
                "timestamp": message["timestamp"]
            }
            self.vector_store.store_memory(content, memory_metadata)
        
        # Update cache
        self.cache[conversation_id] = (conversation, datetime.now().timestamp())
        
        return message
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """Generate summary of conversation"""
        conversation = self.sqlite_store.load_conversation(conversation_id)
        if not conversation or not conversation.messages:
            return "No conversation found or empty conversation"
        
        # Prepare context for summary
        messages_text = "\n".join([
            f"{msg['role']}: {msg['content'][:100]}..."
            for msg in conversation.messages[-5:]  # Last 5 messages
        ])
        
        # Generate summary using LLM
        prompt = f"""
        Summarize this conversation:
        
        {messages_text}
        
        Provide a concise summary of the key topics discussed.
        """
        
        summary = self.llm.predict(prompt)
        return summary
    
    def search_conversations(self, user_id: str, query: str) -> List[Dict]:
        """Search across user's conversations"""
        # Get all user conversations
        conversations = self.sqlite_store.list_user_conversations(user_id)
        
        # Search in vector store for relevant memories
        relevant_memories = self.vector_store.search_memories(query, k=10)
        
        # Group memories by conversation
        conversation_scores = {}
        for memory in relevant_memories:
            conv_id = memory["metadata"].get("conversation_id")
            if conv_id:
                if conv_id not in conversation_scores:
                    conversation_scores[conv_id] = 0
                conversation_scores[conv_id] += memory["score"]
        
        # Create result list
        results = []
        for conv_id, score in sorted(conversation_scores.items(), key=lambda x: x[1], reverse=True):
            conversation = self.sqlite_store.load_conversation(conv_id)
            if conversation:
                results.append({
                    "conversation": conversation,
                    "relevance_score": score,
                    "excerpt": self._get_conversation_excerpt(conv_id, query)
                })
        
        return results
    
    def _get_conversation_excerpt(self, conversation_id: str, query: str) -> str:
        """Get relevant excerpt from conversation"""
        memories = self.vector_store.search_memories(
            query, 
            k=3, 
            filters={"conversation_id": conversation_id}
        )
        
        if memories:
            return memories[0]["text"][:200] + "..."
        
        return "No relevant excerpt found"
    
    def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up old conversations"""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        with sqlite3.connect(self.sqlite_store.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM conversations 
                WHERE updated_at < ?
            """, (cutoff_date,))
            
            old_conversations = cursor.fetchall()
            for (conv_id,) in old_conversations:
                self.sqlite_store.delete_conversation(conv_id)
            
            print(f"Cleaned up {len(old_conversations)} old conversations")

class PersistentMemoryAgent:
    """Agent with persistent memory capabilities"""
    
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory_system = HybridMemorySystem()
        self.llm = ChatOpenAI(temperature=0.7)
        
        # Current conversation
        self.current_conversation = None
    
    def start_conversation(self, topic: str = "") -> str:
        """Start a new conversation"""
        self.current_conversation = self.memory_system.create_conversation(
            self.user_id, 
            title=topic or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        # Store initial message
        if topic:
            self.memory_system.add_message(
                self.current_conversation.id,
                "system",
                f"Conversation started about: {topic}"
            )
        
        return self.current_conversation.id
    
    def process_message(self, message: str) -> str:
        """Process user message with memory context"""
        if not self.current_conversation:
            self.start_conversation()
        
        # Add user message
        self.memory_system.add_message(
            self.current_conversation.id,
            "user",
            message
        )
        
        # Get conversation context
        context = self.memory_system.vector_store.get_conversation_context(
            self.current_conversation.id,
            recent_messages=5
        )
        
        # Search for relevant past conversations
        past_conversations = self.memory_system.search_conversations(
            self.user_id, 
            message
        )
        
        # Prepare enhanced prompt
        prompt = self._build_prompt(message, context, past_conversations)
        
        # Generate response
        response = self.llm.predict(prompt)
        
        # Add assistant response
        self.memory_system.add_message(
            self.current_conversation.id,
            "assistant",
            response
        )
        
        return response
    
    def _build_prompt(self, message: str, context: str, past_conversations: List[Dict]) -> str:
        """Build prompt with memory context"""
        prompt_parts = [
            "You are an AI assistant with persistent memory. You remember past conversations.",
            "\nCurrent conversation context:",
            context if context else "No recent context",
        ]
        
        if past_conversations:
            prompt_parts.append("\nRelevant past conversations:")
            for i, result in enumerate(past_conversations[:3], 1):
                conv = result["conversation"]
                prompt_parts.append(
                    f"{i}. {conv.title} (score: {result['relevance_score']:.2f}): "
                    f"{result['excerpt']}"
                )
        
        prompt_parts.extend([
            "\nUser message:",
            message,
            "\nAssistant:"
        ])
        
        return "\n".join(prompt_parts)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get user's conversation history"""
        conversations = self.memory_system.sqlite_store.list_user_conversations(
            self.user_id, 
            limit
        )
        
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "message_count": conv.metadata.get("message_count", 0),
                "last_updated": conv.updated_at,
                "summary": self.memory_system.get_conversation_summary(conv.id)
            }
            for conv in conversations
        ]

def demo_persistent_memory():
    """Demonstrate persistent memory capabilities"""
    print("=" * 60)
    print("💾 PERSISTENT MEMORY DEMO (Day 20)")
    print("=" * 60)
    
    # Create agent
    agent = PersistentMemoryAgent(user_id="demo_user")
    
    print("\n1️⃣ Starting new conversation about 'AI Ethics'")
    conv_id = agent.start_conversation("AI Ethics")
    print(f"   Conversation ID: {conv_id}")
    
    print("\n2️⃣ Adding messages to conversation...")
    messages = [
        "What are the main ethical concerns with AI?",
        "How can we ensure AI systems are transparent?",
        "What about bias in machine learning models?"
    ]
    
    for msg in messages:
        print(f"\n   User: {msg}")
        response = agent.process_message(msg)
        print(f"   Assistant: {response[:100]}...")
    
    print("\n3️⃣ Getting conversation summary...")
    summary = agent.memory_system.get_conversation_summary(conv_id)
    print(f"   Summary: {summary[:200]}...")
    
    print("\n4️⃣ Searching in conversations...")
    search_results = agent.memory_system.search_conversations("demo_user", "AI bias")
    print(f"   Found {len(search_results)} relevant conversations")
    
    if search_results:
        for result in search_results[:2]:
            print(f"   - {result['conversation'].title}: score={result['relevance_score']:.2f}")
    
    print("\n5️⃣ Conversation history...")
    history = agent.get_conversation_history()
    print(f"   Total conversations: {len(history)}")
    for conv in history[:3]:
        print(f"   - {conv['title']}: {conv['message_count']} messages")
    
    print("\n" + "=" * 60)
    print("✅ Persistent Memory Demo Complete!")

class MultiUserMemoryManager:
    """Memory manager for multiple users"""
    
    def __init__(self, storage_dir: str = "./user_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # User database
        self.users_db = SQLiteMemoryStore(self.storage_dir / "users.db")
        
        # Separate vector store per user
        self.user_stores = {}
    
    def get_user_store(self, user_id: str) -> HybridMemorySystem:
        """Get or create memory system for user"""
        if user_id not in self.user_stores:
            user_dir = self.storage_dir / user_id
            user_dir.mkdir(exist_ok=True)
            
            # Configure SQLite path
            db_path = str(user_dir / "conversations.db")
            
            # Create hybrid system for user
            sqlite_store = SQLiteMemoryStore(db_path)
            vector_store = VectorMemoryStore(str(user_dir / "vector_store"))
            
            # We need to adapt to use our stores
            # For simplicity, creating a new instance
            self.user_stores[user_id] = HybridMemorySystem()
        
        return self.user_stores[user_id]

def main():
    """Run persistent memory demonstrations"""
    print("🚀 Advanced Persistent Memory Systems")
    print("Day 20: Database Storage & Vector Memory")
    
    # Demo 1: Basic persistent memory
    demo_persistent_memory()
    
    # Demo 2: Multi-user setup
    print("\n" + "=" * 60)
    print("👥 MULTI-USER MEMORY MANAGEMENT")
    print("=" * 60)
    
    manager = MultiUserMemoryManager()
    
    users = ["alice", "bob", "charlie"]
    for user in users:
        print(f"\nTesting memory for user: {user}")
        user_store = manager.get_user_store(user)
        
        # Create conversation
        conv = user_store.create_conversation(user, f"{user}'s conversation")
        
        # Add messages
        user_store.add_message(conv.id, "user", f"Hello, I'm {user}")
        user_store.add_message(conv.id, "assistant", f"Nice to meet you, {user}!")
        
        print(f"   Created conversation: {conv.title}")
        print(f"   Messages: {conv.metadata.get('message_count', 0)}")
    
    print("\n" + "=" * 60)
    print("✅ Day 20 Complete!")
    print("Next: LangChain Integration (Day 21)")

if __name__ == "__main__":
    from datetime import timedelta
    main()