import os
# Replace the jwt import with explicit import from PyJWT
import jwt as pyjwt
import bcrypt
from typing import Optional, Dict, List
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

from sqlalchemy import create_engine, Column, Integer, String, func, desc
from sqlalchemy.orm import sessionmaker, declarative_base

# LangChain imports
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.memory import ConversationBufferWindowMemory

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import Base, User, Conversation, Message, Document

# Import knowledge base
from knowledge_base import KnowledgeBase

# ----- Load env vars -----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")  # for demo only
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", "86400"))  # 24 hours

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")

# ----- Database setup -----
# Get database URL from environment or use SQLite as fallback
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./customer_support.db")

# Handle special case for PostgreSQL from Heroku/AWS RDS
if (DATABASE_URL.startswith("postgres://")):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with appropriate connect_args only for SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# ----- FastAPI init -----
app = FastAPI(title="AI Customer Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- LangChain Setup -----
prompt_template = PromptTemplate(
    input_variables=["history", "user_input", "knowledge_base"],
    template=(
        "You are an AI customer support assistant. Be helpful, clear, and professional.\n"
        "If you're unsure about an answer, be honest and suggest escalating to human support.\n\n"
        "Below is information from our knowledge base that might help answer the question:\n"
        "{knowledge_base}\n\n"
        "Conversation history:\n"
        "{history}\n"
        "User: {user_input}\n"
        "AI:"
    )
)

# Use GPT-3.5-turbo-instruct for a good balance of cost and performance
llm = OpenAI(
    openai_api_key=OPENAI_API_KEY, 
    temperature=0.5,
    model_name="gpt-3.5-turbo-instruct"  # Using the more cost-effective model
)
pipeline = prompt_template | llm

# Define path for FAISS index persistence
KNOWLEDGE_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_index.index")

# ----- OAuth2 & JWT -----
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ----- Initialize Knowledge Base -----
kb = KnowledgeBase(index_path=KNOWLEDGE_INDEX_PATH)

# ----- Models / Schemas -----
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None

class UserLogin(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    is_admin: bool
    created_at: datetime

class AnalyticsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    avg_messages_per_conversation: float
    escalation_rate: float
    top_questions: List[Dict]
    daily_usage: List[Dict]
    estimated_input_cost: float
    estimated_output_cost: float
    estimated_total_cost: float

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def decode_token(token: str) -> Dict:
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except pyjwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)) -> User:
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def admin_required(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user


# ----- Routes -----
@app.get("/")
def root():
    return {"message": "AI-Powered Enterprise Customer Support System API", 
            "version": "1.0.0",
            "docs": "/docs"}


@app.post("/register")
def register(user_data: UserCreate, db=Depends(get_db)):
    # Check if user exists
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Check email if provided
    if user_data.email:
        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt())
    
    # Create new user (first user becomes admin)
    is_first_user = db.query(User).count() == 0
    
    new_user = User(
        username=user_data.username, 
        password_hash=hashed.decode("utf-8"),
        email=user_data.email,
        is_admin=is_first_user
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User registered successfully", "is_admin": new_user.is_admin}

@app.post("/login")
def login(user_data: UserLogin, db=Depends(get_db)):
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Check password
    if not bcrypt.checkpw(user_data.password.encode("utf-8"), user.password_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create JWT with expiration
    expiration = datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION)
    payload = {
        "sub": user.username,
        "exp": expiration,
        "is_admin": user.is_admin
    }
    token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
    return {
        "access_token": token, 
        "token_type": "bearer",
        "expires_at": expiration.isoformat(),
        "is_admin": user.is_admin
    }

@app.post("/chat")
def chat(request: ChatRequest, current_user=Depends(get_current_user), db=Depends(get_db)):
    """
    Multi-turn chat endpoint with knowledge base integration
    """
    try:
        user_id = current_user.id
        user_input = request.question.strip()
        
        if not user_input:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Find or create conversation
        active_conversation = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.end_time.is_(None)
        ).order_by(desc(Conversation.start_time)).first()
        
        if not active_conversation:
            # Create a new conversation
            active_conversation = Conversation(user_id=user_id)
            db.add(active_conversation)
            db.commit()
            db.refresh(active_conversation)
        
        # Retrieve conversation history
        history_messages = db.query(Message).filter(
            Message.conversation_id == active_conversation.id
        ).order_by(Message.timestamp).all()
        
        # Format conversation history
        history_formatted = "\n".join([
            f"{'User' if msg.role == 'user' else 'AI'}: {msg.content}"
            for msg in history_messages[-10:]  # Keep last 10 messages for context
        ])
        
        # Query knowledge base for relevant information
        try:
            knowledge_results = kb.search(user_input, top_k=3)
        except Exception as e:
            print(f"Knowledge base search error: {str(e)}")
            knowledge_results = []
            
        # Format knowledge base results
        knowledge_base_text = "\n\n".join([
            f"Document: {result['metadata'].get('file_name', 'Unknown')}\n"
            f"Content: {result['content']}\n"
            f"Relevance: {result['score']:.2f}"
            for result in knowledge_results
        ]) if knowledge_results else "No relevant information found in knowledge base."
        
        # Invoke LLM pipeline
        response_text = pipeline.invoke({
            "history": history_formatted,
            "user_input": user_input,
            "knowledge_base": knowledge_base_text
        })
        
        # Compute confidence score based on knowledge base results
        if knowledge_results:
            # Average the top 3 scores, weighted by position
            weights = [0.6, 0.3, 0.1]
            weighted_scores = [result["score"] * weight for result, weight in zip(knowledge_results, weights)]
            confidence_score = min(100, sum(weighted_scores) / sum(weights[:len(knowledge_results)]) * 100)
        else:
            # Lower confidence when no knowledge base results are found
            confidence_score = 70
        
        # Determine if escalation is needed
        escalate = confidence_score < 70
        
        # Save user message
        user_message = Message(
            conversation_id=active_conversation.id,
            role="user",
            content=user_input
        )
        db.add(user_message)
        
        # Save AI response
        ai_message = Message(
            conversation_id=active_conversation.id,
            role="assistant",
            content=response_text,
            confidence_score=confidence_score,
            was_escalated=escalate,
            used_kb=len(knowledge_results) > 0
        )
        db.add(ai_message)
        
        db.commit()
        
        return {
            "answer": response_text,
            "confidence": confidence_score,
            "escalate": escalate,
            "conversation_id": active_conversation.id
        }
    except Exception as e:
        db.rollback()  # Rollback any pending transaction on error
        print(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ----- Knowledge Base Endpoints -----
@app.post("/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...), 
    current_user: User = Depends(admin_required),
    db = Depends(get_db)
):
    """
    Upload a document to the knowledge base (admin only)
    """
    # Check file size (max 50MB)
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    # Get file extension
    file_name = file.filename
    file_extension = file_name.split(".")[-1].lower() if "." in file_name else ""
    
    # Check supported file types
    supported_types = ["pdf", "txt", "docx", "doc", "xlsx", "xls"]
    if file_extension not in supported_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types: {', '.join(supported_types)}"
        )
    
    # Add to knowledge base
    success = kb.add_document(
        file_content=file_content,
        file_name=file_name,
        file_type=file_extension,
        metadata={"uploader": current_user.username}
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process document")
    
    # Record document in database
    new_doc = Document(
        file_name=file_name,
        file_type=file_extension,
        file_size=file_size,
        uploader_id=current_user.id
    )
    db.add(new_doc)
    db.commit()
    
    return {"message": f"Document {file_name} uploaded successfully"}

@app.get("/knowledge/search")
def search_knowledge_base(
    query: str = Query(..., min_length=3),
    top_k: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_user)
):
    """
    Search the knowledge base
    """
    results = kb.search(query, top_k=top_k)
    
    # Format results for API response
    formatted_results = []
    for result in results:
        formatted_results.append({
            "content": result["content"],
            "file_name": result["metadata"].get("file_name", "Unknown"),
            "relevance_score": result["score"]
        })
    
    return {"results": formatted_results}