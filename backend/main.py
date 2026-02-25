from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import json
import numpy as np
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="RAG Chat Assistant API",
    description="Production-grade RAG-based chat assistant with Gemini AI",
    version="1.0.0"
)

# CORS middleware with environment-based origins
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini client with error handling
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    raise

# In-memory storage (use Redis/DB for production scale)
document_store: List[Dict] = []
sessions: Dict[str, List[Dict]] = {}

# Configuration
MAX_CHUNK_TOKENS = 400
SIMILARITY_THRESHOLD = 0.5
TOP_K_CHUNKS = 3
MAX_HISTORY_PAIRS = 5
MAX_OUTPUT_TOKENS = 300
TEMPERATURE = 0.2

class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=2000)
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace')
        return v.strip()

class ChatResponse(BaseModel):
    reply: str
    tokensUsed: int = 0
    retrievedChunks: int
    similarityScores: List[float] = []

def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Split text into chunks based on sentences.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (approximate)
        
    Returns:
        List of text chunks
    """
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Rough token estimate: ~4 chars per token
        estimated_tokens = (len(current_chunk) + len(sentence)) / 4
        if estimated_tokens < max_tokens:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding vector using Gemini API.
    
    Args:
        text: Input text to embed
        
    Returns:
        Embedding vector as list of floats
        
    Raises:
        Exception: If embedding generation fails
    """
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Similarity score between -1 and 1
    """
    a_np = np.array(a)
    b_np = np.array(b)
    norm_product = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm_product == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / norm_product)

@app.on_event("startup")
async def load_documents():
    """Load and embed documents on startup"""
    try:
        logger.info("Loading documents and generating embeddings...")
        docs_path = Path("../docs.json")
        
        if not docs_path.exists():
            logger.error(f"Documents file not found: {docs_path}")
            raise FileNotFoundError(f"docs.json not found at {docs_path}")
        
        with open(docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        
        for doc in docs:
            chunks = chunk_text(doc["content"])
            for chunk in chunks:
                embedding = get_embedding(chunk)
                document_store.append({
                    "title": doc["title"],
                    "content": chunk,
                    "embedding": embedding
                })
        
        logger.info(f"Successfully loaded {len(document_store)} document chunks")
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise

@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with RAG.
    
    Args:
        request: Chat request with sessionId and message
        
    Returns:
        ChatResponse with reply and metadata
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Initialize session if needed
        if request.sessionId not in sessions:
            sessions[request.sessionId] = []
            logger.info(f"New session created: {request.sessionId}")
        
        # Generate query embedding
        query_embedding = get_embedding(request.message)
        
        # Calculate similarities
        similarities = []
        for doc in document_store:
            sim = cosine_similarity(query_embedding, doc["embedding"])
            similarities.append((sim, doc))
        
        # Sort by similarity and get top K
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_chunks = similarities[:TOP_K_CHUNKS]
        
        # Log for monitoring
        similarity_scores = [score for score, _ in top_chunks]
        logger.info(f"Query: '{request.message[:50]}...' | Top scores: {[f'{s:.3f}' for s in similarity_scores]}")
        
        # Apply threshold
        relevant_chunks = [chunk for score, chunk in top_chunks if score >= SIMILARITY_THRESHOLD]
        
        # Handle no relevant chunks
        if not relevant_chunks:
            logger.warning(f"No relevant chunks found for query (threshold: {SIMILARITY_THRESHOLD})")
            return ChatResponse(
                reply="I don't have enough information to answer that question. Please ask about account management, payments, security, or support topics.",
                tokensUsed=0,
                retrievedChunks=0,
                similarityScores=similarity_scores
            )
        
        # Build context
        context = "\n\n".join([
            f"Document: {chunk['title']}\n{chunk['content']}" 
            for chunk in relevant_chunks
        ])
        
        # Get conversation history
        history = sessions[request.sessionId][-(MAX_HISTORY_PAIRS * 2):]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        # Build prompt
        prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context.

Context:
{context}

Conversation History:
{history_text}

User Question: {request.message}

Instructions:
- Answer based only on the context provided
- Be concise and accurate
- If the context doesn't contain the answer, say "I don't have that information"
- Do not make up information"""

        # Call Gemini LLM
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        )
        
        reply = response.text
        tokens_used = len(prompt.split()) + len(reply.split())
        
        # Store conversation
        sessions[request.sessionId].append({"role": "user", "content": request.message})
        sessions[request.sessionId].append({"role": "assistant", "content": reply})
        
        # Limit history size
        if len(sessions[request.sessionId]) > MAX_HISTORY_PAIRS * 2:
            sessions[request.sessionId] = sessions[request.sessionId][-(MAX_HISTORY_PAIRS * 2):]
        
        logger.info(f"Response generated | Tokens: {tokens_used} | Chunks: {len(relevant_chunks)}")
        
        return ChatResponse(
            reply=reply,
            tokensUsed=tokens_used,
            retrievedChunks=len(relevant_chunks),
            similarityScores=similarity_scores
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred processing your request"
        )

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG Chat API is running",
        "documents": len(document_store),
        "active_sessions": len(sessions)
    }

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "document_chunks": len(document_store),
        "active_sessions": len(sessions),
        "gemini_configured": bool(os.getenv("GOOGLE_API_KEY"))
    }
