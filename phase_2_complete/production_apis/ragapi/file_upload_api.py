"""
File Upload API - Advanced file processing
Supports multiple files, progress tracking, and resumable uploads
"""
import os
import tempfile
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import aiofiles

app = FastAPI(title="Advanced File Upload API")
security = HTTPBearer()

# Models
class UploadStatus(BaseModel):
    upload_id: str = Field(..., description="Unique upload ID")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="Total file size in bytes")
    uploaded_bytes: int = Field(0, description="Bytes uploaded so far")
    status: str = Field("pending", description="Upload status")
    chunks: List[Dict[str, Any]] = Field([], description="Uploaded chunks")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

class ChunkUploadRequest(BaseModel):
    upload_id: str = Field(..., description="Upload ID")
    chunk_index: int = Field(..., description="Chunk index")
    total_chunks: int = Field(..., description="Total number of chunks")
    chunk_data: str = Field(..., description="Base64 encoded chunk data")

# Storage for uploads (use database in production)
uploads_db = {}
uploaded_files_dir = Path("./uploads")
uploaded_files_dir.mkdir(exist_ok=True)

# Dependencies
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "upload-token":
        raise HTTPException(status_code=401, detail="Invalid API token")
    return {"user_id": "upload-user"}

# File processing functions
def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

async def process_uploaded_file(file_path: Path, upload_id: str):
    """Process uploaded file in background"""
    try:
        # Update status to processing
        uploads_db[upload_id].status = "processing"
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        
        # Get file info
        file_size = file_path.stat().st_size
        
        # Here you could:
        # 1. Extract text from various file types
        # 2. Generate thumbnails for images
        # 3. Validate file contents
        # 4. Store in cloud storage
        # 5. Trigger other processing pipelines
        
        # Simulate processing
        import time
        time.sleep(2)
        
        # Update status
        uploads_db[upload_id].status = "completed"
        uploads_db[upload_id].completed_at = datetime.now()
        uploads_db[upload_id].metadata = {
            "file_hash": file_hash,
            "file_size": file_size,
            "processed_at": datetime.now().isoformat()
        }
        
        print(f"✅ File processed: {upload_id}, Hash: {file_hash}")
        
    except Exception as e:
        uploads_db[upload_id].status = "failed"
        uploads_db[upload_id].error = str(e)
        print(f"❌ File processing failed: {upload_id}, Error: {e}")

# Routes
@app.post("/upload/single")
async def upload_single_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    user_info: dict = Depends(verify_token)
):
    """Upload single file with metadata"""
    # Generate upload ID
    upload_id = hashlib.md5(f"{file.filename}{datetime.now()}".encode()).hexdigest()[:16]
    
    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    
    try:
        # Read file content
        content = await file.read()
        
        # Save to temp file
        with open(temp_file.name, "wb") as f:
            f.write(content)
        
        # Create upload record
        upload_status = UploadStatus(
            upload_id=upload_id,
            filename=file.filename,
            file_size=len(content),
            uploaded_bytes=len(content),
            status="uploaded"
        )
        
        # Parse metadata if provided
        if metadata:
            try:
                upload_status.metadata = json.loads(metadata)
            except:
                upload_status.metadata = {"custom": metadata}
        
        uploads_db[upload_id] = upload_status
        
        # Process file in background
        if background_tasks:
            background_tasks.add_task(process_uploaded_file, Path(temp_file.name), upload_id)
        
        return {
            "message": "File uploaded successfully",
            "upload_id": upload_id,
            "filename": file.filename,
            "file_size": len(content),
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload/multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    user_info: dict = Depends(verify_token)
):
    """Upload multiple files simultaneously"""
    results = []
    
    for file in files:
        try:
            # Generate upload ID for each file
            upload_id = hashlib.md5(f"{file.filename}{datetime.now()}".encode()).hexdigest()[:16]
            
            # Read file content
            content = await file.read()
            
            # Save to uploads directory
            file_path = uploaded_files_dir / f"{upload_id}_{file.filename}"
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
            
            # Create upload record
            upload_status = UploadStatus(
                upload_id=upload_id,
                filename=file.filename,
                file_size=len(content),
                uploaded_bytes=len(content),
                status="uploaded"
            )
            
            uploads_db[upload_id] = upload_status
            
            results.append({
                "filename": file.filename,
                "upload_id": upload_id,
                "size": len(content),
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "message": f"Processed {len(files)} files",
        "results": results
    }

@app.post("/upload/chunked/init")
async def init_chunked_upload(
    filename: str = Form(...),
    file_size: int = Form(...),
    chunk_size: int = Form(1024 * 1024),  # 1MB default
    user_info: dict = Depends(verify_token)
):
    """Initialize chunked upload"""
    upload_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:16]
    
    # Calculate total chunks
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    upload_status = UploadStatus(
        upload_id=upload_id,
        filename=filename,
        file_size=file_size,
        status="initialized",
        metadata={
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "chunks_received": 0
        }
    )
    
    uploads_db[upload_id] = upload_status
    
    return {
        "upload_id": upload_id,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "resume_url": f"/upload/chunked/{upload_id}"
    }

@app.post("/upload/chunked/{upload_id}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk: UploadFile = File(...),
    user_info: dict = Depends(verify_token)
):
    """Upload a chunk of file"""
    if upload_id not in uploads_db:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    upload_status = uploads_db[upload_id]
    
    try:
        # Read chunk data
        chunk_data = await chunk.read()
        
        # Save chunk to temporary location
        chunk_dir = uploaded_files_dir / upload_id
        chunk_dir.mkdir(exist_ok=True)
        
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}"
        async with aiofiles.open(chunk_path, "wb") as f:
            await f.write(chunk_data)
        
        # Update upload status
        upload_status.uploaded_bytes += len(chunk_data)
        upload_status.chunks.append({
            "index": chunk_index,
            "size": len(chunk_data),
            "received_at": datetime.now().isoformat()
        })
        
        upload_status.metadata["chunks_received"] = len(upload_status.chunks)
        
        # Check if all chunks received
        if len(upload_status.chunks) == total_chunks:
            upload_status.status = "assembling"
            
            # Assemble file in background
            async def assemble_file():
                try:
                    output_path = uploaded_files_dir / upload_status.filename
                    
                    # Create output file
                    async with aiofiles.open(output_path, "wb") as out_file:
                        # Write chunks in order
                        for i in range(total_chunks):
                            chunk_path = chunk_dir / f"chunk_{i:04d}"
                            if chunk_path.exists():
                                async with aiofiles.open(chunk_path, "rb") as in_file:
                                    chunk_data = await in_file.read()
                                    await out_file.write(chunk_data)
                            
                            # Update progress
                            upload_status.metadata["assembly_progress"] = (i + 1) / total_chunks * 100
                    
                    # Clean up chunks
                    import shutil
                    shutil.rmtree(chunk_dir)
                    
                    # Update final status
                    upload_status.status = "completed"
                    upload_status.completed_at = datetime.now()
                    upload_status.file_size = output_path.stat().st_size
                    
                except Exception as e:
                    upload_status.status = "failed"
                    upload_status.error = str(e)
            
            # Run assembly in background
            import asyncio
            asyncio.create_task(assemble_file())
        
        return {
            "chunk_received": chunk_index,
            "chunks_received": len(upload_status.chunks),
            "total_chunks": total_chunks,
            "uploaded_bytes": upload_status.uploaded_bytes,
            "status": upload_status.status
        }
        
    except Exception as e:
        upload_status.status = "failed"
        upload_status.error = str(e)
        raise HTTPException(status_code=500, detail=f"Chunk upload failed: {str(e)}")

@app.get("/upload/status/{upload_id}")
async def get_upload_status(
    upload_id: str,
    user_info: dict = Depends(verify_token)
):
    """Get upload status"""
    if upload_id not in uploads_db:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return uploads_db[upload_id]

@app.get("/uploads")
async def list_uploads(
    status: Optional[str] = None,
    limit: int = 50,
    user_info: dict = Depends(verify_token)
):
    """List all uploads with optional filtering"""
    uploads = list(uploads_db.values())
    
    if status:
        uploads = [u for u in uploads if u.status == status]
    
    uploads.sort(key=lambda x: x.created_at, reverse=True)
    
    return {
        "count": len(uploads),
        "uploads": uploads[:limit]
    }

@app.delete("/upload/{upload_id}")
async def cancel_upload(
    upload_id: str,
    user_info: dict = Depends(verify_token)
):
    """Cancel and delete an upload"""
    if upload_id not in uploads_db:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    # Clean up files
    upload_status = uploads_db[upload_id]
    
    # Delete assembled file if exists
    file_path = uploaded_files_dir / upload_status.filename
    if file_path.exists():
        file_path.unlink()
    
    # Delete chunk directory if exists
    chunk_dir = uploaded_files_dir / upload_id
    if chunk_dir.exists():
        import shutil
        shutil.rmtree(chunk_dir)
    
    # Remove from database
    del uploads_db[upload_id]
    
    return {"message": f"Upload {upload_id} cancelled and deleted"}

@app.get("/download/{upload_id}")
async def download_file(
    upload_id: str,
    user_info: dict = Depends(verify_token)
):
    """Download uploaded file"""
    if upload_id not in uploads_db:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    upload_status = uploads_db[upload_id]
    
    if upload_status.status != "completed":
        raise HTTPException(status_code=400, detail="File not fully uploaded or processed")
    
    file_path = uploaded_files_dir / upload_status.filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on server")
    
    # Stream file for download
    async def file_streamer():
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(65536):  # 64KB chunks
                yield chunk
    
    headers = {
        "Content-Disposition": f'attachment; filename="{upload_status.filename}"',
        "Content-Type": "application/octet-stream"
    }
    
    return StreamingResponse(
        file_streamer(),
        headers=headers,
        media_type="application/octet-stream"
    )

# File validation endpoints
@app.post("/validate/pdf")
async def validate_pdf(
    file: UploadFile = File(...),
    user_info: dict = Depends(verify_token)
):
    """Validate PDF file"""
    try:
        content = await file.read()
        
        # Check if it's a valid PDF
        if content[:4] != b'%PDF':
            return {
                "valid": False,
                "error": "Not a valid PDF file"
            }
        
        # Additional PDF validation could go here
        # e.g., check for encryption, extract metadata, etc.
        
        return {
            "valid": True,
            "filename": file.filename,
            "size": len(content),
            "page_count": "unknown"  # Would extract with PyPDF2
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

@app.post("/validate/image")
async def validate_image(
    file: UploadFile = File(...),
    user_info: dict = Depends(verify_token)
):
    """Validate image file"""
    try:
        content = await file.read()
        
        # Check common image signatures
        signatures = {
            b'\xff\xd8\xff': 'jpg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif',
            b'BM': 'bmp'
        }
        
        for sig, format in signatures.items():
            if content[:len(sig)] == sig:
                return {
                    "valid": True,
                    "format": format,
                    "filename": file.filename,
                    "size": len(content)
                }
        
        return {
            "valid": False,
            "error": "Not a recognized image format"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

# File statistics
@app.get("/stats")
async def get_upload_stats(user_info: dict = Depends(verify_token)):
    """Get upload statistics"""
    total_uploads = len(uploads_db)
    completed = sum(1 for u in uploads_db.values() if u.status == "completed")
    failed = sum(1 for u in uploads_db.values() if u.status == "failed")
    processing = sum(1 for u in uploads_db.values() if u.status in ["uploaded", "processing"])
    
    total_size = sum(u.file_size for u in uploads_db.values() if u.status == "completed")
    
    return {
        "total_uploads": total_uploads,
        "completed": completed,
        "failed": failed,
        "processing": processing,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }

# Cleanup endpoint (admin)
@app.post("/admin/cleanup")
async def cleanup_old_uploads(
    days_old: int = 7,
    user_info: dict = Depends(verify_token)
):
    """Clean up old uploads"""
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    deleted_count = 0
    deleted_size = 0
    
    for upload_id, upload_status in list(uploads_db.items()):
        if upload_status.created_at < cutoff_date:
            # Delete file if exists
            file_path = uploaded_files_dir / upload_status.filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_path.unlink()
                deleted_size += file_size
            
            # Remove from database
            del uploads_db[upload_id]
            deleted_count += 1
    
    return {
        "message": f"Cleaned up {deleted_count} old uploads",
        "deleted_count": deleted_count,
        "freed_space_bytes": deleted_size,
        "freed_space_mb": round(deleted_size / (1024 * 1024), 2)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 File Upload API starting...")
    print("📁 Upload directory: ./uploads")
    print("🔑 Use Authorization: Bearer upload-token")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)