"""
RAG Document Q&A System - Complete Implementation
A production-ready Retrieval Augmented Generation system for document question answering.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader
)
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
import chromadb
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

# Load environment variables
load_dotenv()

class RAGDocumentQASystem:
    """
    Complete RAG Document Q&A System with advanced features.
    
    Features:
    - Multiple document format support (PDF, TXT, DOCX, MD, CSV)
    - Smart text chunking with overlap
    - Multiple embedding models (OpenAI, HuggingFace)
    - Vector database integration (ChromaDB, FAISS)
    - Conversational memory
    - Source citation
    - Streaming responses
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "openai",  # "openai" or "huggingface"
        model_name: str = "gpt-3.5-turbo",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the RAG Q&A system.
        
        Args:
            persist_directory: Directory to store vector database
            embedding_model: Which embedding model to use
            model_name: OpenAI model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        self.documents = []
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ RAG Document Q&A System initialized")
        print(f"   Embedding Model: {embedding_model}")
        print(f"   LLM Model: {model_name}")
        print(f"   Persist Directory: {persist_directory}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if self.embedding_model == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-ada-002"
            )
        elif self.embedding_model == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from various file formats.
        
        Supported formats:
        - PDF (.pdf)
        - Text (.txt)
        - Markdown (.md)
        - Word (.docx)
        - CSV (.csv)
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of Document objects
        """
        all_documents = []
        
        for file_path in tqdm(file_paths, desc="📚 Loading documents"):
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
            
            file_ext = Path(file_path).suffix.lower()
            
            try:
                if file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_ext == '.md':
                    loader = UnstructuredMarkdownLoader(file_path)
                elif file_ext in ['.doc', '.docx']:
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif file_ext == '.csv':
                    loader = CSVLoader(file_path)
                else:
                    print(f"⚠️  Unsupported file format: {file_ext}")
                    continue
                
                documents = loader.load()
                for doc in documents:
                    doc.metadata["source"] = file_path
                    doc.metadata["loaded_at"] = datetime.now().isoformat()
                
                all_documents.extend(documents)
                print(f"✅ Loaded {len(documents)} pages from {file_path}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {str(e)}")
        
        self.documents = all_documents
        return all_documents
    
    def load_from_folder(self, folder_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a folder.
        
        Args:
            folder_path: Path to folder containing documents
            recursive: Whether to search subfolders recursively
            
        Returns:
            List of Document objects
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Supported extensions
        supported_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx', '.csv'}
        
        # Find all files
        if recursive:
            file_paths = []
            for ext in supported_extensions:
                file_paths.extend(folder_path.rglob(f"*{ext}"))
        else:
            file_paths = []
            for ext in supported_extensions:
                file_paths.extend(folder_path.glob(f"*{ext}"))
        
        file_paths = [str(fp) for fp in file_paths]
        
        return self.load_documents(file_paths)
    
    def load_web_page(self, url: str) -> List[Document]:
        """
        Load content from a web page.
        
        Args:
            url: URL of the web page
            
        Returns:
            List of Document objects
        """
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = url
                doc.metadata["type"] = "web_page"
                doc.metadata["loaded_at"] = datetime.now().isoformat()
            
            self.documents.extend(documents)
            return documents
            
        except Exception as e:
            print(f"❌ Error loading web page {url}: {str(e)}")
            return []
    
    def split_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        """
        Split documents into chunks with overlap.
        
        Args:
            documents: List of documents to split (uses self.documents if None)
            
        Returns:
            List of chunked Document objects
        """
        if documents is None:
            documents = self.documents
        
        if not documents:
            print("⚠️  No documents to split")
            return []
        
        # Use recursive text splitter for better semantic chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = "unknown"
        
        print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(
        self,
        documents: Optional[List[Document]] = None,
        use_chroma: bool = True,
        collection_name: str = "documents"
    ):
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents to index
            use_chroma: Whether to use ChromaDB (True) or FAISS (False)
            collection_name: Name of the collection (for ChromaDB)
            
        Returns:
            Initialized vector store
        """
        if documents is None:
            documents = self.split_documents()
        
        if not documents:
            raise ValueError("No documents to create vector store")
        
        print(f"📊 Creating vector store from {len(documents)} chunks...")
        
        if use_chroma:
            # Use ChromaDB with persistence
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            self.vector_store.persist()
            print(f"✅ ChromaDB vector store created with {len(documents)} chunks")
        else:
            # Use FAISS (in-memory)
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            print(f"✅ FAISS vector store created with {len(documents)} chunks")
        
        return self.vector_store
    
    def load_existing_vector_store(self, collection_name: str = "documents"):
        """
        Load existing vector store from disk.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Loaded vector store or None if not found
        """
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                
                # Get document count
                collection = self.vector_store._collection
                if collection:
                    count = collection.count()
                    print(f"✅ Loaded existing vector store with {count} documents")
                else:
                    print("✅ Loaded existing vector store")
                
                return self.vector_store
            else:
                print("⚠️  No existing vector store found")
                return None
                
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            return None
    
    def initialize_qa_chain(
        self,
        chain_type: str = "stuff",
        return_source_documents: bool = True,
        temperature: float = 0.1,
        streaming: bool = False
    ):
        """
        Initialize QA chain for question answering.
        
        Args:
            chain_type: Type of chain ("stuff", "map_reduce", "refine", "map_rerank")
            return_source_documents: Whether to return source documents
            temperature: LLM temperature
            streaming: Whether to enable streaming responses
            
        Returns:
            Initialized QA chain
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if streaming else []
        )
        
        # Custom prompt template for better answers
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

        Context: {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
            ),
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print(f"✅ QA chain initialized with {chain_type} chain type")
        return self.qa_chain
    
    def initialize_conversational_chain(self, memory_size: int = 5):
        """
        Initialize conversational QA chain with memory.
        
        Args:
            memory_size: Number of previous messages to remember
            
        Returns:
            Initialized conversational chain
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create conversational chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        print(f"✅ Conversational chain initialized with memory size {memory_size}")
        return self.conversation_chain
    
    def ask_question(
        self,
        question: str,
        use_conversation: bool = False,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            use_conversation: Whether to use conversational chain
            return_sources: Whether to include source documents
            
        Returns:
            Dictionary with answer and metadata
        """
        if use_conversation:
            if self.conversation_chain is None:
                self.initialize_conversational_chain()
            
            result = self.conversation_chain({"question": question})
            answer = result["answer"]
            sources = result.get("source_documents", [])
        else:
            if self.qa_chain is None:
                self.initialize_qa_chain()
            
            result = self.qa_chain({"query": question})
            answer = result["result"]
            sources = result.get("source_documents", [])
        
        # Format response
        response = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name
        }
        
        if return_sources and sources:
            response["sources"] = self._format_sources(sources)
            response["source_count"] = len(sources)
        
        return response
    
    def _format_sources(self, source_documents: List[Document]) -> List[Dict]:
        """Format source documents for display"""
        sources = []
        seen_content = set()
        
        for doc in source_documents:
            content_preview = doc.page_content[:200] + "..."
            if content_preview in seen_content:
                continue
            
            seen_content.add(content_preview)
            
            sources.append({
                "content": doc.page_content[:500],  # First 500 chars
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "similarity_score": doc.metadata.get("score", 0.0)
            })
        
        return sources
    
    def semantic_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform semantic search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if filter_metadata:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata
            )
        else:
            results = self.vector_store.similarity_search(query=query, k=k)
        
        return results
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded documents.
        
        Returns:
            Dictionary with document statistics
        """
        if not self.documents:
            return {"total_documents": 0, "total_chunks": 0}
        
        # Count by source
        sources = {}
        for doc in self.documents:
            source = doc.metadata.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        # Calculate total characters
        total_chars = sum(len(doc.page_content) for doc in self.documents)
        
        return {
            "total_documents": len(self.documents),
            "total_characters": total_chars,
            "sources": sources,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
    
    def export_answers(self, questions: List[str], output_file: str = "answers.json"):
        """
        Answer multiple questions and export to JSON.
        
        Args:
            questions: List of questions to answer
            output_file: Output JSON file path
        """
        answers = []
        
        for question in tqdm(questions, desc="🤔 Answering questions"):
            try:
                answer = self.ask_question(question, return_sources=True)
                answers.append(answer)
            except Exception as e:
                print(f"❌ Error answering '{question}': {str(e)}")
                answers.append({
                    "question": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Answers exported to {output_file}")
    
    def interactive_mode(self):
        """Start interactive Q&A session"""
        print("\n" + "="*60)
        print("🤖 RAG Document Q&A Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'stats' to see document statistics")
        print("Type 'clear' to clear conversation memory")
        print("Type 'search <query>' to perform semantic search")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n📝 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    stats = self.get_document_stats()
                    print(f"\n📊 Document Statistics:")
                    print(f"   Total Documents: {stats['total_documents']}")
                    print(f"   Total Characters: {stats['total_characters']:,}")
                    print(f"   Sources: {len(stats['sources'])}")
                    for source, count in stats['sources'].items():
                        print(f"     - {source}: {count} documents")
                    continue
                
                elif user_input.lower() == 'clear':
                    if self.conversation_chain:
                        self.conversation_chain.memory.clear()
                        print("🧹 Conversation memory cleared")
                    else:
                        print("ℹ️  No conversation memory to clear")
                    continue
                
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        print(f"\n🔍 Searching for: {query}")
                        results = self.semantic_search(query, k=3)
                        for i, doc in enumerate(results, 1):
                            print(f"\n📄 Result {i}:")
                            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                            print(f"   Content: {doc.page_content[:300]}...")
                    continue
                
                if not user_input:
                    continue
                
                # Ask the question
                print("\n🤖 Thinking...")
                response = self.ask_question(user_input, use_conversation=True)
                
                print(f"\n💡 Answer: {response['answer']}")
                
                if 'sources' in response:
                    print(f"\n📚 Sources ({response['source_count']}):")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"\n   Source {i}:")
                        print(f"      From: {source['source']}")
                        print(f"      Preview: {source['content'][:150]}...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


# ========== Example Usage Functions ==========

def example_basic_rag():
    """Example of basic RAG usage"""
    print("🧪 Example: Basic RAG Document Q&A")
    print("-" * 40)
    
    # Initialize system
    rag_system = RAGDocumentQASystem(
        persist_directory="./data/chroma_db_basic",
        embedding_model="openai",
        model_name="gpt-3.5-turbo"
    )
    
    # Create sample document
    sample_text = """
    Artificial Intelligence (AI) is the simulation of human intelligence in machines 
    that are programmed to think like humans and mimic their actions. The term may also 
    be applied to any machine that exhibits traits associated with a human mind such 
    as learning and problem-solving.
    
    Machine Learning (ML) is a subset of AI that focuses on building systems that learn 
    or improve performance based on the data they consume. Deep Learning is a subset of 
    ML that uses neural networks with many layers.
    
    Large Language Models (LLMs) like GPT-4 are advanced AI models trained on vast amounts 
    of text data. They can understand, generate, and translate human language with 
    remarkable accuracy.
    
    Vector databases store data as high-dimensional vectors (embeddings) which capture 
    semantic meaning. They enable efficient similarity search for applications like 
    recommendation systems and question answering.
    """
    
    # Save sample text
    sample_file = "./data/sample_ai.txt"
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    # Load and process
    rag_system.load_documents([sample_file])
    rag_system.create_vector_store()
    rag_system.initialize_qa_chain()
    
    # Ask questions
    questions = [
        "What is Artificial Intelligence?",
        "What is the relationship between AI and Machine Learning?",
        "What are vector databases used for?"
    ]
    
    for question in questions:
        print(f"\n📝 Question: {question}")
        answer = rag_system.ask_question(question)
        print(f"🤖 Answer: {answer['answer']}")
        print("-" * 40)

def example_multiple_documents():
    """Example with multiple document types"""
    print("\n🧪 Example: Multiple Document Types")
    print("-" * 40)
    
    # Create sample documents in different formats
    sample_data = {
        "ai_history.txt": """
        AI History:
        1950: Alan Turing proposes the Turing Test
        1956: John McCarthy coins term "Artificial Intelligence"
        1997: IBM Deep Blue beats chess champion Garry Kasparov
        2011: IBM Watson wins Jeopardy!
        2020: GPT-3 released with 175 billion parameters
        """,
        
        "ml_techniques.md": """
        # Machine Learning Techniques
        
        ## Supervised Learning
        - Classification: Categorizing data into classes
        - Regression: Predicting continuous values
        
        ## Unsupervised Learning
        - Clustering: Grouping similar data points
        - Dimensionality Reduction: Reducing feature space
        
        ## Reinforcement Learning
        - Agent learns through rewards/punishments
        - Used in robotics and game playing
        """,
        
        "llm_models.csv": """model,parameters,release_year,organization
        GPT-3,175B,2020,OpenAI
        GPT-4,Unknown,2023,OpenAI
        LLaMA,7B-65B,2023,Meta
        Claude,Unknown,2023,Anthropic
        Bard,137B,2023,Google
        """
    }
    
    # Save sample documents
    for filename, content in sample_data.items():
        filepath = f"./data/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if filename.endswith('.csv'):
            # Parse and save as proper CSV
            import csv
            from io import StringIO
            
            f = StringIO(content)
            reader = csv.DictReader(f)
            rows = list(reader)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    # Initialize and process
    rag_system = RAGDocumentQASystem(
        persist_directory="./data/chroma_db_multi",
        embedding_model="huggingface"
    )
    
    # Load all documents from folder
    documents = rag_system.load_from_folder("./data", recursive=False)
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create vector store
    rag_system.create_vector_store()
    
    # Initialize QA chain
    rag_system.initialize_qa_chain(chain_type="map_reduce")
    
    # Test questions
    test_questions = [
        "When was the term 'Artificial Intelligence' coined?",
        "What are the main types of Machine Learning?",
        "Which LLM has the most parameters?",
        "What happened in AI in 1997?"
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        answer = rag_system.ask_question(question, return_sources=True)
        print(f"🤖 Answer: {answer['answer']}")
        
        if answer.get('sources'):
            print(f"📚 Sources used: {len(answer['sources'])}")
        print("-" * 40)

def example_web_scraping():
    """Example with web page scraping"""
    print("\n🧪 Example: Web Page RAG")
    print("-" * 40)
    
    # Note: WebBaseLoader requires additional dependencies
    # pip install nest-asyncio beautifulsoup4
    
    rag_system = RAGDocumentQASystem(
        persist_directory="./data/chroma_db_web",
        model_name="gpt-4"
    )
    
    # Load web pages (example URLs)
    example_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        # Add more URLs as needed
    ]
    
    print("⚠️  Web scraping example disabled to avoid network calls")
    print("   Uncomment the code and install required packages to test")
    
    # Uncomment to test:
    # for url in example_urls:
    #     documents = rag_system.load_web_page(url)
    #     print(f"✅ Loaded {len(documents)} documents from {url}")
    
    # rag_system.create_vector_store()
    # rag_system.initialize_qa_chain()
    
    # question = "What is artificial intelligence according to Wikipedia?"
    # answer = rag_system.ask_question(question)
    # print(f"\n📝 Question: {question}")
    # print(f"🤖 Answer: {answer['answer'][:500]}...")

# ========== Main Execution ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Document Q&A System")
    parser.add_argument("--mode", choices=["basic", "multi", "web", "interactive"], 
                       default="basic", help="Example mode to run")
    parser.add_argument("--folder", type=str, help="Folder with documents to load")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--questions-file", type=str, help="File with list of questions (JSON or txt)")
    parser.add_argument("--output", type=str, default="answers.json", help="Output file for answers")
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key-here")
    
    # Run examples based on mode
    if args.mode == "basic":
        example_basic_rag()
    
    elif args.mode == "multi":
        example_multiple_documents()
    
    elif args.mode == "web":
        example_web_scraping()
    
    elif args.mode == "interactive":
        if args.folder:
            # Interactive mode with custom documents
            rag_system = RAGDocumentQASystem(
                persist_directory="./data/chroma_db_interactive",
                model_name="gpt-4"
            )
            
            print(f"📂 Loading documents from: {args.folder}")
            rag_system.load_from_folder(args.folder)
            rag_system.create_vector_store()
            rag_system.initialize_conversational_chain()
            
            if args.question:
                # Answer single question
                answer = rag_system.ask_question(args.question, use_conversation=True)
                print(f"\n📝 Question: {args.question}")
                print(f"🤖 Answer: {answer['answer']}")
                
                if answer.get('sources'):
                    print(f"\n📚 Sources ({answer['source_count']}):")
                    for i, source in enumerate(answer['sources'], 1):
                        print(f"   {i}. {source['source']}")
            else:
                # Start interactive session
                rag_system.interactive_mode()
        else:
            print("❌ Please provide a folder with --folder for interactive mode")
    
    # Handle single question
    if args.question and args.mode != "interactive":
        rag_system = RAGDocumentQASystem()
        
        if args.folder:
            rag_system.load_from_folder(args.folder)
            rag_system.create_vector_store()
            rag_system.initialize_qa_chain()
        
        answer = rag_system.ask_question(args.question, return_sources=True)
        
        print(f"\n📝 Question: {answer['question']}")
        print(f"🤖 Answer: {answer['answer']}")
        
        if answer.get('sources'):
            print(f"\n📚 Sources used:")
            for i, source in enumerate(answer['sources'], 1):
                print(f"\n   Source {i}:")
                print(f"      File: {source['source']}")
                print(f"      Content: {source['content'][:150]}...")
    
    # Handle batch questions
    if args.questions_file and os.path.exists(args.questions_file):
        rag_system = RAGDocumentQASystem()
        
        if args.folder:
            rag_system.load_from_folder(args.folder)
            rag_system.create_vector_store()
            rag_system.initialize_qa_chain()
        
        # Load questions
        if args.questions_file.endswith('.json'):
            import json
            with open(args.questions_file, 'r') as f:
                questions = json.load(f)
        else:
            with open(args.questions_file, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
        
        rag_system.export_answers(questions, args.output)