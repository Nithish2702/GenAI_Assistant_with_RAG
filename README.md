# RAG Chat Assistant

A production-grade GenAI-powered chat assistant using Retrieval-Augmented Generation (RAG) with React frontend and FastAPI backend.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React     │─────▶│   FastAPI    │─────▶│   Gemini    │
│  Frontend   │      │   Backend    │      │     API     │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  In-Memory   │
                     │Vector Store  │
                     └──────────────┘
```

## RAG Workflow

1. **Document Loading**: Load documents from `docs.json`
2. **Chunking**: Split documents into 300-500 token chunks
3. **Embedding Generation**: Generate embeddings using Gemini `gemini-embedding-001`
4. **Storage**: Store chunks with embeddings in memory
5. **Query Processing**:
   - User sends question
   - Generate query embedding
   - Calculate cosine similarity with all document embeddings
   - Retrieve top 3 chunks above threshold (0.5)
6. **Context Injection**: Build prompt with retrieved chunks + conversation history
7. **LLM Response**: Send to Gemini 2.0 Flash with temperature 0.2
8. **Return**: Send response with metadata to frontend

## Embedding Strategy

- **Model**: Gemini `gemini-embedding-001` (768 dimensions)
- **Chunking**: Sentence-based splitting, ~400 tokens per chunk
- **Similarity**: Cosine similarity for retrieval
- **Threshold**: 0.5 minimum similarity score
- **Top-K**: Retrieve 3 most relevant chunks

## Similarity Search

Uses cosine similarity formula:
```
similarity = dot(A, B) / (||A|| * ||B||)
```

Where:
- A = query embedding vector
- B = document embedding vector
- Result ranges from -1 to 1 (higher = more similar)

## Prompt Design

```
Context: [Retrieved document chunks]
Conversation History: [Last 5 exchanges]
User Question: [Current question]
Instructions: Answer based ONLY on context
```

**Rationale**:
- Low temperature (0.2) for factual responses
- Explicit grounding instruction prevents hallucination
- Conversation history maintains context
- Fallback response when similarity too low

## Production Features

### Backend
- ✅ Comprehensive logging with timestamps
- ✅ Input validation with Pydantic
- ✅ Error handling with proper HTTP status codes
- ✅ Environment-based configuration
- ✅ Health check endpoints
- ✅ API documentation (FastAPI auto-docs)
- ✅ Type hints throughout
- ✅ Docstrings for all functions
- ✅ Configuration constants
- ✅ Graceful error messages

### Frontend
- ✅ Chat history persistence (localStorage)
- ✅ Session management
- ✅ Loading states
- ✅ Error handling
- ✅ Responsive design
- ✅ Markdown rendering
- ✅ Timestamps
- ✅ Token usage tracking
- ✅ Clean UI with dark theme

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- Google Gemini API key

### Backend Setup

1. Navigate to backend:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

5. Add your Google API key to `.env`:
```
GOOGLE_API_KEY=your-key-here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

6. Run server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

Frontend runs at `http://localhost:5173`

## API Endpoints

### POST /api/chat

**Request**:
```json
{
  "sessionId": "session_123",
  "message": "How do I reset my password?"
}
```

**Response**:
```json
{
  "reply": "Users can reset their password by navigating to Settings > Security > Reset Password...",
  "tokensUsed": 120,
  "retrievedChunks": 3,
  "similarityScores": [0.89, 0.82, 0.75]
}
```

### GET /

Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "message": "RAG Chat API is running",
  "documents": 45,
  "active_sessions": 3
}
```

### GET /health

Detailed health check

**Response**:
```json
{
  "status": "healthy",
  "document_chunks": 45,
  "active_sessions": 3,
  "gemini_configured": true
}
```

## Production Deployment

### Backend Deployment

1. **Environment Variables**:
```bash
GOOGLE_API_KEY=your-production-key
ALLOWED_ORIGINS=https://yourdomain.com
```

2. **Run with Gunicorn** (production WSGI server):
```bash
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

3. **Docker** (optional):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Frontend Deployment

1. **Build for production**:
```bash
npm run build
```

2. **Deploy** to:
   - Vercel: `vercel deploy`
   - Netlify: `netlify deploy --prod`
   - Static hosting: Upload `dist/` folder

3. **Update API URL** in production:
```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
```

## Features

✅ Real embedding-based retrieval (not keyword search)  
✅ Cosine similarity ranking  
✅ Similarity threshold filtering  
✅ Context-aware responses  
✅ Conversation history (last 5 pairs)  
✅ Session management with persistence  
✅ Loading states  
✅ Markdown rendering  
✅ Timestamps  
✅ Token usage tracking  
✅ Error handling  
✅ Input validation  
✅ Logging and monitoring  
✅ Health check endpoints  
✅ Production-ready code structure  

## Tech Stack

- **Frontend**: React 18, Vite, React Markdown
- **Backend**: FastAPI, Python 3.9+, Pydantic
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Google Gemini embedding-001
- **Vector Operations**: NumPy
- **Storage**: In-memory (Python dict)

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI server with RAG logic
│   ├── requirements.txt     # Python dependencies
│   ├── .env                 # Environment variables
│   └── .env.example         # Environment template
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main chat component
│   │   ├── App.css          # Styles
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Global styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── docs.json                # Knowledge base documents
├── .gitignore
└── README.md
```

## Monitoring & Logging

The backend includes comprehensive logging:
- Request/response logging
- Similarity score tracking
- Error logging with stack traces
- Session creation tracking
- Performance metrics (token usage)

View logs in console or configure file logging for production.

## Security Considerations

- API keys stored in environment variables
- Input validation on all endpoints
- CORS configured for specific origins
- Error messages don't expose sensitive info
- Rate limiting recommended for production

## Scaling Considerations

For production scale:
- Replace in-memory storage with Redis/PostgreSQL
- Add caching layer for embeddings
- Implement rate limiting
- Use load balancer for multiple instances
- Add monitoring (Prometheus/Grafana)
- Implement proper session management (Redis)

## License

MIT
