"""
Multi-Document Assistant - Advanced RAG System
Handles multiple document collections with sophisticated query routing and answer synthesis.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import hashlib
import warnings
warnings.filterwarnings('ignore')

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader
)
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document, BaseRetriever
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.tools import BaseTool
from langchain.utilities import GoogleSearchAPIWrapper
import openai
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load environment variables
load_dotenv()

# ========== Enums and Data Classes ==========

class DocumentCollection(Enum):
    """Types of document collections"""
    GENERAL = "general"
    TECHNICAL = "technical"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    RESEARCH = "research"
    PERSONAL = "personal"
    ARCHIVE = "archive"

class QueryType(Enum):
    """Types of queries for routing"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    LEGAL = "legal"
    MEDICAL = "medical"
    UNKNOWN = "unknown"

@dataclass
class DocumentSet:
    """A set of related documents"""
    name: str
    collection_type: DocumentCollection
    documents: List[Document]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        self.id = hashlib.md5(f"{self.name}_{self.collection_type.value}".encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "collection_type": self.collection_type.value,
            "document_count": len(self.documents),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

@dataclass
class AssistantResponse:
    """Enhanced response from multi-document assistant"""
    answer: str
    query: str
    query_type: QueryType
    collections_used: List[str]
    documents_consulted: int
    confidence: float
    citations: List[Dict[str, Any]]
    processing_time_ms: float
    model: str
    cost_estimate: float = 0.0
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "query": self.query,
            "query_type": self.query_type.value,
            "collections_used": self.collections_used,
            "documents_consulted": self.documents_consulted,
            "confidence": round(self.confidence, 3),
            "citations": self.citations,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model": self.model,
            "cost_estimate": round(self.cost_estimate, 4),
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }
    
    def format_for_display(self) -> str:
        """Format response for display"""
        output = f"🤖 Answer: {self.answer}\n\n"
        
        if self.citations:
            output += "📚 Sources:\n"
            for i, citation in enumerate(self.citations, 1):
                output += f"  {i}. {citation.get('source', 'Unknown')}\n"
                if citation.get('page'):
                    output += f"     Page: {citation['page']}\n"
                if citation.get('confidence'):
                    output += f"     Confidence: {citation['confidence']:.2%}\n"
        
        output += f"\n📊 Metadata:\n"
        output += f"  Query Type: {self.query_type.value}\n"
        output += f"  Collections Used: {', '.join(self.collections_used)}\n"
        output += f"  Documents Consulted: {self.documents_consulted}\n"
        output += f"  Confidence: {self.confidence:.2%}\n"
        output += f"  Processing Time: {self.processing_time_ms:.0f}ms\n"
        
        if self.warnings:
            output += f"\n⚠️  Warnings:\n"
            for warning in self.warnings:
                output += f"  • {warning}\n"
        
        if self.suggestions:
            output += f"\n💡 Suggestions:\n"
            for suggestion in self.suggestions:
                output += f"  • {suggestion}\n"
        
        return output

# ========== Query Router ==========

class QueryRouter:
    """Routes queries to appropriate document collections"""
    
    def __init__(self):
        # Query type patterns
        self.patterns = {
            QueryType.FACTUAL: [
                "what is", "who is", "when did", "where is",
                "define", "explain", "describe"
            ],
            QueryType.ANALYTICAL: [
                "analyze", "evaluate", "assess", "critique",
                "pros and cons", "advantages disadvantages"
            ],
            QueryType.SUMMARIZATION: [
                "summarize", "overview", "brief", "key points",
                "main ideas", "tl;dr"
            ],
            QueryType.COMPARISON: [
                "compare", "contrast", "difference between",
                "similarities", "vs", "versus"
            ],
            QueryType.CREATIVE: [
                "create", "write", "generate", "imagine",
                "story", "poem", "essay"
            ],
            QueryType.TECHNICAL: [
                "code", "algorithm", "implementation",
                "technical", "programming", "debug"
            ],
            QueryType.LEGAL: [
                "legal", "law", "contract", "agreement",
                "compliance", "regulation"
            ],
            QueryType.MEDICAL: [
                "medical", "health", "treatment", "diagnosis",
                "symptoms", "medicine"
            ]
        }
        
        # Collection mapping
        self.collection_mapping = {
            QueryType.TECHNICAL: [DocumentCollection.TECHNICAL, DocumentCollection.RESEARCH],
            QueryType.LEGAL: [DocumentCollection.LEGAL],
            QueryType.MEDICAL: [DocumentCollection.MEDICAL],
            QueryType.FINANCIAL: [DocumentCollection.FINANCIAL],
            QueryType.RESEARCH: [DocumentCollection.RESEARCH],
            QueryType.FACTUAL: [DocumentCollection.GENERAL],
            QueryType.ANALYTICAL: [DocumentCollection.RESEARCH, DocumentCollection.GENERAL],
            QueryType.SUMMARIZATION: [DocumentCollection.GENERAL],
            QueryType.COMPARISON: [DocumentCollection.RESEARCH],
            QueryType.CREATIVE: [DocumentCollection.PERSONAL]
        }
    
    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify query type with confidence.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (query_type, confidence)
        """
        query_lower = query.lower()
        
        # Check for exact matches
        scores = {}
        for qtype, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    score += 1
            
            # Also check for keywords in query
            if qtype.value in query_lower:
                score += 2
            
            scores[qtype] = score
        
        # Find best match
        best_type = max(scores.items(), key=lambda x: x[1])
        
        if best_type[1] > 0:
            confidence = min(best_type[1] / 3, 1.0)  # Normalize confidence
            return best_type[0], confidence
        else:
            return QueryType.UNKNOWN, 0.5
    
    def route_to_collections(
        self,
        query: str,
        available_collections: List[DocumentCollection]
    ) -> List[DocumentCollection]:
        """
        Route query to appropriate collections.
        
        Args:
            query: User query
            available_collections: List of available collections
            
        Returns:
            List of collections to query
        """
        query_type, confidence = self.classify_query(query)
        
        # Get suggested collections for this query type
        suggested = self.collection_mapping.get(query_type, [DocumentCollection.GENERAL])
        
        # Filter to only available collections
        filtered = [col for col in suggested if col in available_collections]
        
        # If no specific collections found, use general
        if not filtered and DocumentCollection.GENERAL in available_collections:
            filtered = [DocumentCollection.GENERAL]
        
        # Also consider all collections if query is broad
        if query_type == QueryType.UNKNOWN or len(query.split()) < 3:
            filtered = available_collections[:3]  # Limit to first 3
        
        return filtered
    
    def get_query_analysis(self, query: str) -> Dict[str, Any]:
        """Get detailed analysis of query"""
        query_type, confidence = self.classify_query(query)
        keywords = self._extract_keywords(query)
        
        return {
            "query": query,
            "query_type": query_type.value,
            "confidence": confidence,
            "keywords": keywords,
            "word_count": len(query.split()),
            "complexity": self._assess_complexity(query)
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        
        if word_count < 3:
            return "simple"
        elif word_count < 8:
            return "medium"
        else:
            return "complex"

# ========== Multi-Collection Retriever ==========

class MultiCollectionRetriever(BaseRetriever):
    """Retriever that searches across multiple collections"""
    
    def __init__(
        self,
        collections: Dict[str, Chroma],
        weights: Optional[Dict[str, float]] = None,
        k_per_collection: int = 3,
        max_total: int = 10
    ):
        super().__init__()
        self.collections = collections
        self.weights = weights or {name: 1.0 for name in collections.keys()}
        self.k_per_collection = k_per_collection
        self.max_total = max_total
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents from all collections"""
        all_documents = []
        
        for collection_name, vector_store in self.collections.items():
            try:
                # Get documents from this collection
                docs = vector_store.similarity_search(
                    query=query,
                    k=self.k_per_collection
                )
                
                # Apply collection weight to scores
                weight = self.weights.get(collection_name, 0.5)
                for doc in docs:
                    # Store collection info in metadata
                    doc.metadata["collection"] = collection_name
                    doc.metadata["collection_weight"] = weight
                
                all_documents.extend(docs)
                
            except Exception as e:
                print(f"⚠️  Error retrieving from {collection_name}: {e}")
        
        # Sort by relevance (simplified - in production use actual scores)
        # For now, use collection weights and position
        def sort_key(doc):
            weight = doc.metadata.get("collection_weight", 0.5)
            # Simulate score based on weight
            return weight * (1.0 - (all_documents.index(doc) / len(all_documents)))
        
        all_documents.sort(key=sort_key, reverse=True)
        
        return all_documents[:self.max_total]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents"""
        # For simplicity, use sync version
        return self.get_relevant_documents(query)

# ========== Answer Synthesizer ==========

class AnswerSynthesizer:
    """Synthesizes answers from multiple document sources"""
    
    def __init__(self, llm_model: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Different prompts for different query types
        self.prompts = {
            QueryType.FACTUAL: self._create_factual_prompt(),
            QueryType.ANALYTICAL: self._create_analytical_prompt(),
            QueryType.SUMMARIZATION: self._create_summarization_prompt(),
            QueryType.COMPARISON: self._create_comparison_prompt(),
            QueryType.CREATIVE: self._create_creative_prompt(),
            QueryType.TECHNICAL: self._create_technical_prompt()
        }
        
        # Default prompt
        self.default_prompt = self._create_default_prompt()
    
    def _create_default_prompt(self) -> PromptTemplate:
        """Create default prompt template"""
        template = """You are a helpful AI assistant with access to multiple document collections.
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Guidelines:
        1. Answer based only on the provided context
        2. If the context doesn't contain the answer, say so
        3. Cite specific sources when providing facts
        4. Be concise and accurate
        
        Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_factual_prompt(self) -> PromptTemplate:
        """Create prompt for factual questions"""
        template = """You are answering a factual question. Use the provided documents to give accurate, specific answers.
        
        Documents:
        {context}
        
        Question: {question}
        
        Requirements:
        1. Provide specific facts, numbers, or dates when available
        2. Cite which document each fact comes from
        3. If different documents contradict, note this
        4. If information is incomplete, say so
        
        Factual Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_analytical_prompt(self) -> PromptTemplate:
        """Create prompt for analytical questions"""
        template = """You are performing an analysis. Use the documents to provide a thoughtful, balanced analysis.
        
        Documents:
        {context}
        
        Question: {question}
        
        Analysis Guidelines:
        1. Identify key themes and patterns
        2. Consider multiple perspectives
        3. Evaluate evidence quality
        4. Draw reasonable conclusions
        5. Note limitations or gaps
        
        Analytical Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_summarization_prompt(self) -> PromptTemplate:
        """Create prompt for summarization"""
        template = """You are creating a summary. Synthesize information from multiple documents into a coherent summary.
        
        Documents:
        {context}
        
        Request: {question}
        
        Summary Guidelines:
        1. Include main points from each relevant document
        2. Organize information logically
        3. Omit minor details
        4. Maintain original meaning
        5. Keep it concise
        
        Summary:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_comparison_prompt(self) -> PromptTemplate:
        """Create prompt for comparison questions"""
        template = """You are comparing different topics or perspectives. Use the documents to highlight similarities and differences.
        
        Documents:
        {context}
        
        Question: {question}
        
        Comparison Guidelines:
        1. Identify key comparison points
        2. Present similarities clearly
        3. Present differences clearly
        4. Use specific examples from documents
        5. Provide balanced perspective
        
        Comparison Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_creative_prompt(self) -> PromptTemplate:
        """Create prompt for creative tasks"""
        template = """You are assisting with a creative task. Use the documents as inspiration or reference material.
        
        Documents (for inspiration):
        {context}
        
        Request: {question}
        
        Creative Guidelines:
        1. Use document content as inspiration
        2. Be creative but stay relevant
        3. Incorporate elements from documents if appropriate
        4. Add your own creative elements
        5. Have fun with it!
        
        Creative Response:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_technical_prompt(self) -> PromptTemplate:
        """Create prompt for technical questions"""
        template = """You are answering a technical question. Provide precise, accurate technical information.
        
        Technical Documents:
        {context}
        
        Question: {question}
        
        Technical Guidelines:
        1. Be precise with technical details
        2. Include code or formulas if relevant
        3. Explain technical concepts clearly
        4. Note technical limitations or considerations
        5. Cite technical sources
        
        Technical Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def synthesize_answer(
        self,
        query: str,
        documents: List[Document],
        query_type: QueryType = QueryType.UNKNOWN
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Synthesize answer from multiple documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            query_type: Type of query
            
        Returns:
            Tuple of (answer, citations, confidence)
        """
        if not documents:
            return "I don't have enough information to answer this question.", [], 0.0
        
        # Prepare context
        context_parts = []
        citations = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"Document {i}")
            collection = doc.metadata.get("collection", "Unknown")
            page = doc.metadata.get("page", "")
            
            context_parts.append(f"[Document {i} from {collection}]\n{doc.page_content}\n")
            
            citations.append({
                "document_id": i,
                "source": source,
                "collection": collection,
                "page": page,
                "content_preview": doc.page_content[:100] + "...",
                "confidence": doc.metadata.get("score", 0.5)
            })
        
        context = "\n---\n".join(context_parts)
        
        # Select appropriate prompt
        prompt = self.prompts.get(query_type, self.default_prompt)
        
        try:
            # Get answer from LLM
            with get_openai_callback() as cb:
                chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=None,  # We're providing context directly
                    chain_type_kwargs={"prompt": prompt}
                )
                
                # Create fake retriever output
                fake_docs = [Document(page_content=context, metadata={})]
                
                result = chain({
                    "query": query,
                    "input_documents": fake_docs
                })
                
                answer = result["result"]
                
                # Calculate confidence based on document relevance
                confidences = [c["confidence"] for c in citations]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                
                # Adjust confidence based on answer length and specificity
                answer_words = len(answer.split())
                confidence = avg_confidence * min(answer_words / 50, 1.0)  # Longer answers more confident
                
                return answer, citations, min(confidence, 1.0)
                
        except Exception as e:
            print(f"❌ Error synthesizing answer: {e}")
            return f"Error generating answer: {str(e)}", citations, 0.0

# ========== Multi-Document Assistant ==========

class MultiDocumentAssistant:
    """
    Advanced assistant that handles multiple document collections.
    
    Features:
    - Multiple document collections with different types
    - Intelligent query routing
    - Cross-collection retrieval
    - Advanced answer synthesis
    - Conversation memory
    - Performance monitoring
    """
    
    def __init__(
        self,
        persist_base: str = "./data/multi_doc_assistant",
        llm_model: str = "gpt-4",
        embedding_model: str = "openai"
    ):
        """
        Initialize multi-document assistant.
        
        Args:
            persist_base: Base directory for persistence
            llm_model: OpenAI model to use
            embedding_model: Embedding model ("openai" or "huggingface")
        """
        self.persist_base = persist_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Initialize components
        self.embeddings = self._initialize_embeddings()
        self.query_router = QueryRouter()
        self.answer_synthesizer = AnswerSynthesizer(llm_model)
        
        # State
        self.collections: Dict[str, DocumentSet] = {}
        self.vector_stores: Dict[str, Chroma] = {}
        self.multi_retriever: Optional[MultiCollectionRetriever] = None
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Statistics
        self.stats = {
            "queries_processed": 0,
            "documents_loaded": 0,
            "collections_created": 0,
            "total_processing_time_ms": 0
        }
        
        # Create base directory
        Path(persist_base).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Multi-Document Assistant initialized")
        print(f"   Model: {llm_model}")
        print(f"   Embeddings: {embedding_model}")
        print(f"   Persist Base: {persist_base}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if self.embedding_model == "openai":
            return OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002"
            )
        else:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def create_collection(
        self,
        name: str,
        collection_type: DocumentCollection,
        documents: List[Document],
        metadata: Optional[Dict] = None
    ) -> DocumentSet:
        """
        Create a new document collection.
        
        Args:
            name: Collection name
            collection_type: Type of collection
            documents: List of documents
            metadata: Additional metadata
            
        Returns:
            Created DocumentSet
        """
        # Create document set
        doc_set = DocumentSet(
            name=name,
            collection_type=collection_type,
            documents=documents,
            metadata=metadata or {}
        )
        
        # Store document set
        self.collections[doc_set.id] = doc_set
        
        # Create vector store
        persist_dir = os.path.join(self.persist_base, doc_set.id)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name=doc_set.id
        )
        vector_store.persist()
        
        # Store vector store
        self.vector_stores[doc_set.id] = vector_store
        
        # Update multi-retriever
        self._update_multi_retriever()
        
        # Update statistics
        self.stats["collections_created"] += 1
        self.stats["documents_loaded"] += len(documents)
        
        print(f"✅ Created collection: {name} ({doc_set.id})")
        print(f"   Type: {collection_type.value}")
        print(f"   Documents: {len(documents)}")
        
        return doc_set
    
    def load_collection_from_folder(
        self,
        name: str,
        collection_type: DocumentCollection,
        folder_path: str,
        metadata: Optional[Dict] = None
    ) -> DocumentSet:
        """
        Load documents from folder into a collection.
        
        Args:
            name: Collection name
            collection_type: Type of collection
            folder_path: Path to folder with documents
            metadata: Additional metadata
            
        Returns:
            Created DocumentSet
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find all supported files
        supported_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx', '.csv'}
        file_paths = []
        
        for ext in supported_extensions:
            file_paths.extend(folder_path.glob(f"*{ext}"))
            file_paths.extend(folder_path.glob(f"*{ext.upper()}"))
        
        # Load documents
        documents = []
        for file_path in tqdm(file_paths, desc=f"📚 Loading {name}"):
            try:
                file_ext = file_path.suffix.lower()
                
                if file_ext == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                elif file_ext == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                elif file_ext == '.csv':
                    loader = CSVLoader(str(file_path))
                elif file_ext == '.md':
                    loader = UnstructuredMarkdownLoader(str(file_path))
                elif file_ext in ['.doc', '.docx']:
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                else:
                    continue
                
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["collection"] = name
                
                documents.extend(loaded_docs)
                
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
        
        if not documents:
            raise ValueError(f"No documents loaded from {folder_path}")
        
        # Create collection
        collection_metadata = {
            "source_folder": str(folder_path),
            "file_count": len(file_paths),
            **(metadata or {})
        }
        
        return self.create_collection(name, collection_type, documents, collection_metadata)
    
    def _update_multi_retriever(self):
        """Update multi-collection retriever with current collections"""
        if not self.vector_stores:
            self.multi_retriever = None
            return
        
        # Calculate weights based on collection type
        weights = {}
        for doc_set in self.collections.values():
            # Assign higher weight to more specific collections
            if doc_set.collection_type == DocumentCollection.TECHNICAL:
                weights[doc_set.id] = 1.5
            elif doc_set.collection_type == DocumentCollection.RESEARCH:
                weights[doc_set.id] = 1.3
            elif doc_set.collection_type == DocumentCollection.GENERAL:
                weights[doc_set.id] = 1.0
            else:
                weights[doc_set.id] = 1.2
        
        self.multi_retriever = MultiCollectionRetriever(
            collections=self.vector_stores,
            weights=weights,
            k_per_collection=3,
            max_total=15
        )
    
    def ask(
        self,
        query: str,
        use_conversation: bool = False,
        specific_collections: Optional[List[str]] = None
    ) -> AssistantResponse:
        """
        Ask a question to the assistant.
        
        Args:
            query: User question
            use_conversation: Whether to use conversation memory
            specific_collections: Specific collections to query (None for automatic)
            
        Returns:
            AssistantResponse with answer and metadata
        """
        import time
        start_time = time.time()
        
        # Update statistics
        self.stats["queries_processed"] += 1
        
        # Analyze query
        query_analysis = self.query_router.get_query_analysis(query)
        query_type = QueryType(query_analysis["query_type"])
        
        # Determine which collections to query
        if specific_collections:
            # Use specified collections
            collections_to_query = [
                col_id for col_id in specific_collections 
                if col_id in self.collections
            ]
        else:
            # Automatic routing
            available_collections = list(self.collections.keys())
            suggested = self.query_router.route_to_collections(query, [
                doc_set.collection_type 
                for doc_set in self.collections.values()
            ])
            
            # Map collection types to collection IDs
            collections_to_query = []
            for col_type in suggested:
                for col_id, doc_set in self.collections.items():
                    if doc_set.collection_type == col_type:
                        collections_to_query.append(col_id)
        
        if not collections_to_query:
            # No collections available
            processing_time = (time.time() - start_time) * 1000
            self.stats["total_processing_time_ms"] += processing_time
            
            return AssistantResponse(
                answer="I don't have any documents to reference for this question.",
                query=query,
                query_type=query_type,
                collections_used=[],
                documents_consulted=0,
                confidence=0.0,
                citations=[],
                processing_time_ms=processing_time,
                model=self.llm_model,
                warnings=["No document collections available"]
            )
        
        # Retrieve documents
        retrieved_documents = []
        
        if self.multi_retriever and not specific_collections:
            # Use multi-retriever for cross-collection search
            retrieved_documents = self.multi_retriever.get_relevant_documents(query)
        else:
            # Search specific collections
            for col_id in collections_to_query:
                if col_id in self.vector_stores:
                    try:
                        docs = self.vector_stores[col_id].similarity_search(
                            query=query,
                            k=3
                        )
                        for doc in docs:
                            doc.metadata["collection_id"] = col_id
                            doc.metadata["collection_name"] = self.collections[col_id].name
                        
                        retrieved_documents.extend(docs)
                    except Exception as e:
                        print(f"⚠️  Error searching collection {col_id}: {e}")
        
        # Remove duplicates (by content hash)
        unique_documents = []
        seen_hashes = set()
        
        for doc in retrieved_documents:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(doc)
        
        # Limit to top documents
        unique_documents = unique_documents[:10]
        
        # Synthesize answer
        answer, citations, confidence = self.answer_synthesizer.synthesize_answer(
            query=query,
            documents=unique_documents,
            query_type=query_type
        )
        
        # Update conversation memory if requested
        if use_conversation:
            self.conversation_memory.save_context(
                {"input": query},
                {"output": answer}
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        self.stats["total_processing_time_ms"] += processing_time
        
        # Get collection names used
        collection_names = []
        for doc in unique_documents:
            col_name = doc.metadata.get("collection_name", "Unknown")
            if col_name not in collection_names:
                collection_names.append(col_name)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query, unique_documents, confidence)
        
        # Generate warnings if needed
        warnings = []
        if confidence < 0.3:
            warnings.append("Low confidence answer - may be inaccurate")
        if len(unique_documents) < 2:
            warnings.append("Limited documents consulted")
        
        # Create response
        response = AssistantResponse(
            answer=answer,
            query=query,
            query_type=query_type,
            collections_used=collection_names,
            documents_consulted=len(unique_documents),
            confidence=confidence,
            citations=citations,
            processing_time_ms=processing_time,
            model=self.llm_model,
            suggestions=suggestions,
            warnings=warnings
        )
        
        return response
    
    def _generate_suggestions(
        self,
        query: str,
        documents: List[Document],
        confidence: float
    ) -> List[str]:
        """Generate suggestions based on query and results"""
        suggestions = []
        
        if confidence < 0.5:
            suggestions.append("Try rephrasing your question for better results")
        
        if len(documents) < 3:
            suggestions.append("Consider adding more documents to relevant collections")
        
        # Check query complexity
        word_count = len(query.split())
        if word_count < 3:
            suggestions.append("Try asking a more specific question")
        elif word_count > 20:
            suggestions.append("Consider breaking your question into smaller parts")
        
        # Check for follow-up questions
        if "compare" in query.lower():
            suggestions.append("You might want to ask about specific differences")
        elif "how to" in query.lower():
            suggestions.append("Consider asking for step-by-step instructions")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def get_collection_info(self) -> List[Dict[str, Any]]:
        """Get information about all collections"""
        info = []
        
        for doc_set in self.collections.values():
            # Get document count from vector store
            doc_count = 0
            if doc_set.id in self.vector_stores:
                try:
                    collection = self.vector_stores[doc_set.id]._collection
                    if collection:
                        doc_count = collection.count()
                except:
                    doc_count = len(doc_set.documents)
            
            info.append({
                "id": doc_set.id,
                "name": doc_set.name,
                "type": doc_set.collection_type.value,
                "document_count": doc_count,
                "created_at": doc_set.created_at,
                "updated_at": doc_set.updated_at,
                "metadata": doc_set.metadata
            })
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assistant statistics"""
        avg_processing_time = 0
        if self.stats["queries_processed"] > 0:
            avg_processing_time = (
                self.stats["total_processing_time_ms"] / 
                self.stats["queries_processed"]
            )
        
        return {
            **self.stats,
            "collections_count": len(self.collections),
            "vector_stores_count": len(self.vector_stores),
            "avg_processing_time_ms": avg_processing_time,
            "memory_size": len(self.conversation_memory.buffer) if hasattr(self.conversation_memory, 'buffer') else 0
        }
    
    def clear_conversation_memory(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()
        print("🧹 Conversation memory cleared")
    
    def export_conversation(self, output_file: str = "conversation.json"):
        """Export conversation history"""
        try:
            history = self.conversation_memory.load_memory_variables({})
            with open(output_file, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"✅ Conversation exported to {output_file}")
        except Exception as e:
            print(f"❌ Error exporting conversation: {e}")
    
    def interactive_mode(self):
        """Start interactive assistant session"""
        print("\n" + "=" * 60)
        print("🤖 Multi-Document Assistant - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  'collections' - List available collections")
        print("  'stats' - Show statistics")
        print("  'clear' - Clear conversation memory")
        print("  'export' - Export conversation")
        print("  'quit' - Exit")
        print("=" * 60)
        
        use_conversation = True
        
        while True:
            try:
                user_input = input("\n📝 Your question: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'collections':
                    collections = self.get_collection_info()
                    print(f"\n📚 Available Collections ({len(collections)}):")
                    for col in collections:
                        print(f"\n  {col['name']} ({col['id']})")
                        print(f"    Type: {col['type']}")
                        print(f"    Documents: {col['document_count']}")
                        print(f"    Created: {col['created_at']}")
                    continue
                
                elif user_input.lower() == 'stats':
                    stats = self.get_statistics()
                    print(f"\n📊 Statistics:")
                    print(f"  Queries Processed: {stats['queries_processed']}")
                    print(f"  Collections: {stats['collections_count']}")
                    print(f"  Documents Loaded: {stats['documents_loaded']}")
                    print(f"  Avg Processing Time: {stats['avg_processing_time_ms']:.0f}ms")
                    continue
                
                elif user_input.lower() == 'clear':
                    self.clear_conversation_memory()
                    continue
                
                elif user_input.lower() == 'export':
                    self.export_conversation()
                    continue
                
                if not user_input:
                    continue
                
                # Check for collection-specific query
                specific_collections = None
                if user_input.startswith('@'):
                    parts = user_input[1:].split(' ', 1)
                    if len(parts) == 2:
                        collection_name, query = parts
                        # Find collection ID by name
                        for col_id, doc_set in self.collections.items():
                            if doc_set.name.lower() == collection_name.lower():
                                specific_collections = [col_id]
                                break
                        
                        if not specific_collections:
                            print(f"⚠️  Collection '{collection_name}' not found")
                            continue
                
                # Ask question
                print("\n🤖 Thinking...")
                response = self.ask(
                    query=query,
                    use_conversation=use_conversation,
                    specific_collections=specific_collections
                )
                
                # Display response
                print("\n" + response.format_for_display())
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

# ========== Example Usage ==========

def create_sample_collections(assistant: MultiDocumentAssistant):
    """Create sample collections for demonstration"""
    print("\n📚 Creating sample collections...")
    
    # Sample technical documents
    tech_docs = [
        Document(
            page_content="""
            Python is a high-level programming language known for its simplicity.
            It supports multiple programming paradigms including object-oriented and functional programming.
            Python is widely used in data science, web development, and automation.
            """,
            metadata={"source": "python_basics.txt", "category": "programming"}
        ),
        Document(
            page_content="""
            Machine learning algorithms enable computers to learn from data.
            Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.
            Deep learning uses neural networks with many layers.
            """,
            metadata={"source": "ml_overview.txt", "category": "ai"}
        )
    ]
    
    # Sample research documents
    research_docs = [
        Document(
            page_content="""
            Artificial Intelligence research focuses on creating intelligent machines.
            Key areas include natural language processing, computer vision, and robotics.
            Recent advances in large language models have transformed the field.
            """,
            metadata={"source": "ai_research.pdf", "category": "research"}
        ),
        Document(
            page_content="""
            Vector databases store embeddings for efficient similarity search.
            They enable applications like recommendation systems and semantic search.
            Popular options include Pinecone, Weaviate, and ChromaDB.
            """,
            metadata={"source": "vector_db_research.pdf", "category": "database"}
        )
    ]
    
    # Sample general documents
    general_docs = [
        Document(
            page_content="""
            Effective communication is essential in business and personal relationships.
            Clear writing helps convey ideas effectively and avoid misunderstandings.
            Practice and feedback improve communication skills over time.
            """,
            metadata={"source": "communication_guide.txt", "category": "general"}
        )
    ]
    
    # Create collections
    assistant.create_collection(
        name="Technical Documentation",
        collection_type=DocumentCollection.TECHNICAL,
        documents=tech_docs,
        metadata={"version": "1.0", "maintainer": "AI Team"}
    )
    
    assistant.create_collection(
        name="Research Papers",
        collection_type=DocumentCollection.RESEARCH,
        documents=research_docs,
        metadata={"domain": "Computer Science", "year": "2024"}
    )
    
    assistant.create_collection(
        name="General Knowledge",
        collection_type=DocumentCollection.GENERAL,
        documents=general_docs,
        metadata={"scope": "broad", "language": "English"}
    )
    
    print(f"✅ Created {len(assistant.collections)} sample collections")

def demo_multi_doc_assistant():
    """Demonstrate multi-document assistant"""
    print("=" * 60)
    print("🧪 DEMO: Multi-Document Assistant")
    print("=" * 60)
    
    # Initialize assistant
    assistant = MultiDocumentAssistant(
        persist_base="./data/multi_doc_demo",
        llm_model="gpt-3.5-turbo"
    )
    
    # Create sample collections
    create_sample_collections(assistant)
    
    # Test queries
    test_queries = [
        ("What is Python programming language?", "factual"),
        ("Compare machine learning and deep learning", "comparison"),
        ("Summarize the key points about artificial intelligence", "summarization"),
        ("How do vector databases work?", "technical")
    ]
    
    print("\n🧪 Testing assistant with different query types:")
    print("-" * 60)
    
    for query, expected_type in test_queries:
        print(f"\n📝 Query: {query}")
        print(f"   Expected type: {expected_type}")
        
        response = assistant.ask(query)
        
        print(f"\n🤖 Answer: {response.answer[:200]}...")
        print(f"📊 Query Type: {response.query_type.value}")
        print(f"📚 Collections Used: {', '.join(response.collections_used)}")
        print(f"📄 Documents Consulted: {response.documents_consulted}")
        print(f"🎯 Confidence: {response.confidence:.2%}")
        
        if response.citations:
            print(f"🔗 Sources: {len(response.citations)}")
            for i, citation in enumerate(response.citations[:2], 1):
                print(f"   {i}. {citation['source']} (Confidence: {citation['confidence']:.2%})")
        
        print("-" * 40)
    
    # Test conversation memory
    print("\n💬 Testing conversation memory:")
    print("-" * 40)
    
    conversation_queries = [
        "What programming languages are mentioned?",
        "Which one is best for beginners?"
    ]
    
    for query in conversation_queries:
        print(f"\n📝 Query: {query}")
        response = assistant.ask(query, use_conversation=True)
        print(f"🤖 Answer: {response.answer[:150]}...")
    
    # Show statistics
    print("\n📊 Assistant Statistics:")
    stats = assistant.get_statistics()
    print(f"  Queries Processed: {stats['queries_processed']}")
    print(f"  Collections: {stats['collections_count']}")
    print(f"  Documents: {stats['documents_loaded']}")
    print(f"  Avg Processing Time: {stats['avg_processing_time_ms']:.0f}ms")
    
    # Collection info
    print("\n📚 Collection Information:")
    collections = assistant.get_collection_info()
    for col in collections:
        print(f"  • {col['name']}: {col['document_count']} documents ({col['type']})")
    
    print("\n✅ Demo completed!")
    print("=" * 60)

def load_real_documents():
    """Load real documents for more realistic testing"""
    print("\n📚 Loading real documents for advanced testing...")
    
    # Create sample files in a data directory
    data_dir = "./data/real_documents"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    sample_files = {
        "technology.txt": """
        Artificial Intelligence Trends 2024
        
        1. Generative AI continues to dominate with models like GPT-4 and Claude
        2. Multimodal AI combines text, image, and audio understanding
        3. AI ethics and safety become critical concerns
        4. Edge AI brings intelligence to devices without cloud connectivity
        5. AI-assisted programming tools improve developer productivity
        
        Key Technologies: Transformers, Diffusion Models, Neural Networks
        """,
        
        "business.txt": """
        Business Strategy for AI Startups
        
        Market Analysis:
        - Global AI market expected to reach $1.8 trillion by 2030
        - Enterprise AI adoption growing at 35% CAGR
        - Top sectors: Healthcare, Finance, Retail, Manufacturing
        
        Competitive Advantage:
        1. Proprietary data sets
        2. Specialized domain expertise
        3. Scalable infrastructure
        4. Strong partnerships
        
        Revenue Models:
        - SaaS subscriptions
        - API usage fees
        - Custom development
        - Consulting services
        """,
        
        "research.md": """
        # Research Paper Summary: Advanced RAG Systems
        
        ## Abstract
        Retrieval-Augmented Generation systems have evolved significantly. This paper presents novel approaches for multi-document RAG with improved accuracy and efficiency.
        
        ## Key Contributions
        1. Hybrid retrieval combining semantic and keyword search
        2. Dynamic query routing based on document collections
        3. Confidence scoring for answer reliability
        4. Cross-document citation generation
        
        ## Results
        - 45% improvement over baseline RAG systems
        - 30% reduction in hallucination rate
        - 60% faster response times with caching
        
        ## Conclusion
        Multi-document RAG systems show significant promise for enterprise knowledge management.
        """
    }
    
    # Save files
    for filename, content in sample_files.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return data_dir

def advanced_demo():
    """Advanced demo with real document loading"""
    print("=" * 60)
    print("🚀 ADVANCED DEMO: Multi-Document Assistant")
    print("=" * 60)
    
    # Initialize assistant
    assistant = MultiDocumentAssistant(
        persist_base="./data/advanced_demo",
        llm_model="gpt-4"
    )
    
    # Load real documents
    data_dir = load_real_documents()
    
    # Create collections from folders
    print("\n📚 Creating collections from real documents...")
    
    # Technology collection
    tech_dir = os.path.join(data_dir)
    assistant.load_collection_from_folder(
        name="Technology News",
        collection_type=DocumentCollection.TECHNICAL,
        folder_path=tech_dir,
        metadata={"year": "2024", "topic": "AI Trends"}
    )
    
    # Business collection
    assistant.load_collection_from_folder(
        name="Business Strategies",
        collection_type=DocumentCollection.FINANCIAL,
        folder_path=tech_dir,  # Using same directory for demo
        metadata={"industry": "Technology", "audience": "Entrepreneurs"}
    )
    
    # Research collection
    assistant.load_collection_from_folder(
        name="Research Papers",
        collection_type=DocumentCollection.RESEARCH,
        folder_path=tech_dir,
        metadata={"academic": "true", "peer_reviewed": "simulated"}
    )
    
    print(f"\n✅ Created {len(assistant.collections)} collections")
    
    # Interactive testing
    print("\n🧪 Interactive Testing Mode")
    print("Type your questions or 'quit' to exit")
    print("-" * 60)
    
    test_questions = [
        "What are the key AI trends for 2024?",
        "How should AI startups approach business strategy?",
        "What improvements do advanced RAG systems offer?",
        "Compare technology trends with business strategies"
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        
        # Get detailed query analysis
        analysis = assistant.query_router.get_query_analysis(question)
        print(f"   Query Type: {analysis['query_type']}")
        print(f"   Keywords: {', '.join(analysis['keywords'])}")
        print(f"   Complexity: {analysis['complexity']}")
        
        # Get answer
        response = assistant.ask(question, use_conversation=True)
        
        print(f"\n🤖 Answer: {response.answer}")
        print(f"📊 Confidence: {response.confidence:.2%}")
        print(f"📚 Collections Used: {', '.join(response.collections_used)}")
        
        if response.suggestions:
            print(f"💡 Suggestions: {', '.join(response.suggestions)}")
        
        print("-" * 40)
    
    # Show advanced statistics
    print("\n📈 Advanced Statistics:")
    stats = assistant.get_statistics()
    
    print(f"  Total Queries: {stats['queries_processed']}")
    print(f"  Total Processing Time: {stats['total_processing_time_ms']:.0f}ms")
    print(f"  Average per Query: {stats['avg_processing_time_ms']:.0f}ms")
    
    # Collection performance (simulated)
    print("\n🎯 Collection Performance:")
    collections = assistant.get_collection_info()
    for col in collections:
        # Simulate some metrics
        relevance_score = 0.7 + (hash(col['name']) % 30) / 100  # Random between 0.7-1.0
        usage_count = (hash(col['name']) % 5) + 1  # Random 1-5
        
        print(f"  {col['name']}:")
        print(f"    Documents: {col['document_count']}")
        print(f"    Relevance Score: {relevance_score:.2%}")
        print(f"    Usage Count: {usage_count}")
    
    # Export conversation
    assistant.export_conversation("advanced_demo_conversation.json")
    
    print("\n✅ Advanced demo completed!")
    print("=" * 60)

# ========== Main Execution ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Document Assistant")
    parser.add_argument("--mode", choices=["demo", "advanced", "interactive"], 
                       default="demo", help="Mode to run")
    parser.add_argument("--folder", type=str, help="Folder with documents to load")
    parser.add_argument("--collection", type=str, help="Specific collection to query")
    parser.add_argument("--query", type=str, help="Single query to test")
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or create a .env file")
        exit(1)
    
    # Run based on mode
    if args.mode == "demo":
        demo_multi_doc_assistant()
    
    elif args.mode == "advanced":
        advanced_demo()
    
    elif args.mode == "interactive":
        # Initialize assistant
        assistant = MultiDocumentAssistant()
        
        # Load collections if folder specified
        if args.folder and os.path.exists(args.folder):
            print(f"📂 Loading documents from: {args.folder}")
            
            # Create collections based on subfolders
            import glob
            for subfolder in glob.glob(os.path.join(args.folder, "*")):
                if os.path.isdir(subfolder):
                    folder_name = os.path.basename(subfolder)
                    collection_type = DocumentCollection.GENERAL
                    
                    # Guess collection type from folder name
                    if any(word in folder_name.lower() for word in ["tech", "code", "programming"]):
                        collection_type = DocumentCollection.TECHNICAL
                    elif any(word in folder_name.lower() for word in ["research", "paper", "study"]):
                        collection_type = DocumentCollection.RESEARCH
                    elif any(word in folder_name.lower() for word in ["legal", "law", "contract"]):
                        collection_type = DocumentCollection.LEGAL
                    
                    try:
                        assistant.load_collection_from_folder(
                            name=folder_name,
                            collection_type=collection_type,
                            folder_path=subfolder
                        )
                    except Exception as e:
                        print(f"⚠️  Error loading {subfolder}: {e}")
        else:
            # Create sample collections
            create_sample_collections(assistant)
        
        # Start interactive mode
        assistant.interactive_mode()
    
    # Handle single query
    if args.query:
        assistant = MultiDocumentAssistant()
        
        # Load sample collections
        create_sample_collections(assistant)
        
        # Specific collection if specified
        specific_collections = None
        if args.collection:
            # Find collection by name
            for col_id, doc_set in assistant.collections.items():
                if doc_set.name.lower() == args.collection.lower():
                    specific_collections = [col_id]
                    break
        
        # Ask question
        response = assistant.ask(args.query, specific_collections=specific_collections)
        
        print(f"\n📝 Query: {response.query}")
        print(f"🔧 Query Type: {response.query_type.value}")
        print(f"🤖 Answer: {response.answer}")
        
        if response.collections_used:
            print(f"📚 Collections Used: {', '.join(response.collections_used)}")
        
        print(f"📊 Confidence: {response.confidence:.2%}")
        print(f"⏱️  Processing Time: {response.processing_time_ms:.0f}ms")
        
        if response.citations:
            print(f"\n🔗 Sources ({len(response.citations)}):")
            for i, citation in enumerate(response.citations, 1):
                print(f"  {i}. {citation['source']}")
                if citation.get('confidence'):
                    print(f"     Confidence: {citation['confidence']:.2%}")
        
        # Export response
        with open("single_query_response.json", "w") as f:
            json.dump(response.to_dict(), f, indent=2)
        
        print(f"\n✅ Response saved to: single_query_response.json")