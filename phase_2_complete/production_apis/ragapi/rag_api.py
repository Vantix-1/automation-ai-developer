"""
RAG System API - Day 40-42
Production-ready Retrieval-Augmented Generation API
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Production-ready Retrieval-Augmented Generation API with document processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    collection_id: Optional[str] = Field(None, description="Collection ID to search in")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    include_sources: bool = Field(True, description="Include source documents in response")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field([], description="Source documents")
    processing_time: float = Field(..., description="Processing time in seconds")
    collection_id: str = Field(..., description="Collection used")

class DocumentInfo(BaseModel):
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(..., description="Number of text chunks")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    collection_id: str = Field(..., description="Collection ID")

class CollectionInfo(BaseModel):
    id: str = Field(..., description="Collection ID")
    name: str = Field(..., description="Collection name")
    document_count: int = Field(..., description="Number of documents")
    chunk_count: int = Field(..., description="Total chunks in collection")
    created_at: datetime = Field(..., description="Creation timestamp")

# Global state
vector_stores = {}  # collection_id -> Chroma vector store
documents_db = {}   # collection_id -> List[DocumentInfo]
collections_db = {}  # collection_id -> CollectionInfo
embeddings = None
llm = None

# Initialize AI models
def initialize_models():
    """Initialize embeddings and LLM models"""
    global embeddings, llm
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
        # Fallback to local embeddings
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode
            print("✅ Using local sentence transformer embeddings")
        except:
            print("❌ No embeddings model available")
            embeddings = None
    else:
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        print("✅ Using OpenAI embeddings")
    
    # Initialize LLM
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        print("✅ LLM initialized")
    except:
        print("⚠️  LLM initialization failed")

# Document processing
def process_document(file_path: str, filename: str) -> List[Document]:
    """Process document and split into chunks"""
    # Determine loader based on file extension
    if filename.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif filename.lower().endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        # Try text loader for other files
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "filename": filename,
            "total_chunks": len(chunks)
        })
    
    return chunks

# RAG prompt template
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context.

Context information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided documents."
3. Cite specific parts of the context when relevant
4. Keep your answer concise and focused

Answer:
"""

rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Dependencies
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    # Simple token validation - replace with proper auth in production
    if token != "rag-api-token":
        raise HTTPException(status_code=401, detail="Invalid API token")
    return {"user_id": "api-user"}

# Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RAG System API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation API",
        "endpoints": {
            "/query": "POST - Query documents",
            "/upload": "POST - Upload documents",
            "/collections": "GET - List collections",
            "/collections/{id}": "GET - Get collection",
            "/documents": "GET - List documents",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "embeddings": "available" if embeddings else "unavailable",
            "llm": "available" if llm else "unavailable",
            "vector_stores": len(vector_stores)
        }
    }
    return status

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user_info: dict = Depends(verify_token)
):
    """Query documents using RAG"""
    import time
    start_time = time.time()
    
    if not embeddings:
        raise HTTPException(status_code=500, detail="Embeddings model not available")
    
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not available")
    
    # Determine which collection to use
    collection_id = request.collection_id
    if not collection_id:
        # Use default collection or first available
        if vector_stores:
            collection_id = list(vector_stores.keys())[0]
        else:
            raise HTTPException(status_code=400, detail="No collections available")
    
    if collection_id not in vector_stores:
        raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
    
    vector_store = vector_stores[collection_id]
    
    # Perform similarity search
    docs = vector_store.similarity_search(
        request.query,
        k=request.top_k
    )
    
    # Prepare context
    context = "\n\n".join([
        f"Source {i+1} (from {doc.metadata.get('filename', 'unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])
    
    # Generate answer using LLM
    try:
        chain = rag_prompt | llm
        answer = chain.invoke({"context": context, "question": request.query})
        
        # Extract answer content
        if hasattr(answer, 'content'):
            answer_text = answer.content
        else:
            answer_text = str(answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    processing_time = time.time() - start_time
    
    # Prepare sources
    sources = []
    if request.include_sources:
        for doc in docs:
            sources.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0)  # Add similarity score if available
            })
    
    return QueryResponse(
        answer=answer_text,
        sources=sources,
        processing_time=round(processing_time, 3),
        collection_id=collection_id
    )

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
    user_info: dict = Depends(verify_token)
):
    """Upload and process document"""
    if not embeddings:
        raise HTTPException(status_code=500, detail="Embeddings model not available")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.md', '.csv']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed: {allowed_extensions}"
        )
    
    # Generate collection ID
    collection_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, collection_name))
    
    # Create collection if it doesn't exist
    if collection_id not in collections_db:
        collections_db[collection_id] = CollectionInfo(
            id=collection_id,
            name=collection_name,
            document_count=0,
            chunk_count=0,
            created_at=datetime.now()
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Process document
        chunks = process_document(tmp_file_path, file.filename)
        
        # Initialize or get vector store
        if collection_id not in vector_stores:
            # Create new vector store
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=f"./chroma_db/{collection_id}"
            )
            vector_stores[collection_id] = vector_store
        else:
            vector_store = vector_stores[collection_id]
        
        # Add documents to vector store
        vector_store.add_documents(chunks)
        
        # Create document info
        doc_id = str(uuid.uuid4())
        document_info = DocumentInfo(
            id=doc_id,
            filename=file.filename,
            file_size=len(content),
            chunk_count=len(chunks),
            uploaded_at=datetime.now(),
            collection_id=collection_id
        )
        
        # Update collections database
        if collection_id not in documents_db:
            documents_db[collection_id] = []
        
        documents_db[collection_id].append(document_info)
        
        # Update collection stats
        collection = collections_db[collection_id]
        collection.document_count += 1
        collection.chunk_count += len(chunks)
        
        # Persist vector store
        vector_store.persist()
        
        return {
            "message": "Document uploaded successfully",
            "document_id": doc_id,
            "collection_id": collection_id,
            "chunks_created": len(chunks),
            "file_size": len(content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

@app.get("/collections")
async def list_collections(user_info: dict = Depends(verify_token)):
    """List all collections"""
    collections = list(collections_db.values())
    return {
        "count": len(collections),
        "collections": collections
    }

@app.get("/collections/{collection_id}")
async def get_collection(
    collection_id: str,
    user_info: dict = Depends(verify_token)
):
    """Get collection details"""
    if collection_id not in collections_db:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    collection = collections_db[collection_id]
    
    # Get documents in this collection
    collection_docs = documents_db.get(collection_id, [])
    
    return {
        **collection.dict(),
        "documents": collection_docs
    }

@app.get("/documents")
async def list_documents(
    collection_id: Optional[str] = None,
    user_info: dict = Depends(verify_token)
):
    """List documents"""
    if collection_id:
        if collection_id not in documents_db:
            raise HTTPException(status_code=404, detail="Collection not found")
        docs = documents_db[collection_id]
    else:
        # Get all documents
        docs = []
        for collection_docs in documents_db.values():
            docs.extend(collection_docs)
    
    return {
        "count": len(docs),
        "documents": docs
    }

@app.delete("/collections/{collection_id}")
async def delete_collection(
    collection_id: str,
    user_info: dict = Depends(verify_token)
):
    """Delete collection and all associated documents"""
    if collection_id not in collections_db:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Remove from databases
    del collections_db[collection_id]
    
    if collection_id in documents_db:
        del documents_db[collection_id]
    
    if collection_id in vector_stores:
        # Clean up vector store files
        vector_store = vector_stores[collection_id]
        try:
            vector_store.delete_collection()
        except:
            pass
        del vector_stores[collection_id]
    
    # Clean up persistence directory
    persist_dir = Path(f"./chroma_db/{collection_id}")
    if persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)
    
    return {"message": f"Collection {collection_id} deleted successfully"}

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user_info: dict = Depends(verify_token)
):
    """Delete specific document"""
    # Find document in all collections
    for collection_id, docs in documents_db.items():
        for i, doc in enumerate(docs):
            if doc.id == document_id:
                # Remove document from list
                removed_doc = docs.pop(i)
                
                # Update collection stats
                collection = collections_db[collection_id]
                collection.document_count -= 1
                collection.chunk_count -= removed_doc.chunk_count
                
                # Note: In production, you'd also need to remove embeddings from vector store
                # This is a simplified version
                
                return {"message": f"Document {document_id} deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Document not found")

# Search endpoints
@app.post("/search")
async def semantic_search(
    query: str,
    collection_id: Optional[str] = None,
    top_k: int = 5,
    user_info: dict = Depends(verify_token)
):
    """Semantic search in documents"""
    if not embeddings:
        raise HTTPException(status_code=500, detail="Embeddings model not available")
    
    if collection_id and collection_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Search in all collections or specific collection
    if collection_id:
        collections_to_search = [collection_id]
    else:
        collections_to_search = list(vector_stores.keys())
    
    results = []
    for coll_id in collections_to_search:
        vector_store = vector_stores[coll_id]
        docs = vector_store.similarity_search(query, k=top_k)
        
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "collection_id": coll_id,
                "collection_name": collections_db[coll_id].name if coll_id in collections_db else "unknown"
            })
    
    # Sort results by relevance (simplified)
    results.sort(key=lambda x: len(x["content"]), reverse=True)
    
    return {
        "query": query,
        "count": len(results),
        "results": results[:top_k]
    }

# Admin endpoints
@app.post("/admin/reindex")
async def reindex_collection(
    collection_id: str,
    user_info: dict = Depends(verify_token)
):
    """Reindex collection (admin only)"""
    # In production, add admin role check
    if collection_id not in collections_db:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Reindexing logic would go here
    # This is a placeholder for actual reindexing implementation
    
    return {
        "message": f"Reindexing triggered for collection {collection_id}",
        "collection_id": collection_id
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 RAG System API starting up...")
    print(f"📅 Started at: {datetime.now().isoformat()}")
    
    # Create chroma_db directory
    Path("./chroma_db").mkdir(exist_ok=True)
    
    # Initialize AI models
    initialize_models()
    
    # Load existing collections from disk
    load_existing_collections()
    
    print(f"✅ Loaded {len(collections_db)} collections")
    print("🔑 API token: rag-api-token")
    print("📚 Upload documents via POST /upload")
    print("❓ Query via POST /query")

def load_existing_collections():
    """Load existing collections from disk"""
    chroma_dir = Path("./chroma_db")
    if not chroma_dir.exists():
        return
    
    for collection_dir in chroma_dir.iterdir():
        if collection_dir.is_dir():
            collection_id = collection_dir.name
            collection_name = f"loaded_{collection_id}"
            
            if embeddings:
                try:
                    vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        persist_directory=str(collection_dir)
                    )
                    vector_stores[collection_id] = vector_store
                    
                    # Estimate document count (simplified)
                    # In production, load actual metadata
                    collections_db[collection_id] = CollectionInfo(
                        id=collection_id,
                        name=collection_name,
                        document_count=1,  # Placeholder
                        chunk_count=100,   # Placeholder
                        created_at=datetime.now()
                    )
                    
                    print(f"  Loaded collection: {collection_id}")
                    
                except Exception as e:
                    print(f"  Error loading collection {collection_id}: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting RAG System API...")
    print("📚 OpenAPI docs: http://localhost:8000/docs")
    print("🔐 Use Authorization: Bearer rag-api-token")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )