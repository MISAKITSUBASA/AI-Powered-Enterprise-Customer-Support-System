import os
# Update JWT import statement
import jwt
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

# Alternative import approach
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import Base, User, Conversation, Message, Document

# Import knowledge base
from knowledge_base import KnowledgeBase
from env_utils import get_openai_api_key, print_masked_key

# Add this near the top of your file, after the imports
import logging
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai.error import RateLimitError, APIConnectionError, APIError

# Configure logging to show more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Get a logger for your application
logger = logging.getLogger("app")

# ----- Load env vars -----
load_dotenv()
OPENAI_API_KEY = get_openai_api_key()
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")  # for demo only
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", "86400"))  # 24 hours

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
else:
    print_masked_key(OPENAI_API_KEY, "Using OpenAI API Key")

# ----- FastAPI init -----
app = FastAPI(title="AI Customer Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ----- OAuth2 & JWT -----
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def decode_token(token: str) -> Dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
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

# Define path for FAISS index persistence
KNOWLEDGE_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_index.index")

# ----- Initialize Knowledge Base -----
kb = KnowledgeBase(index_path=KNOWLEDGE_INDEX_PATH)

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
    model_name="gpt-3.5-turbo-instruct"
)
pipeline = prompt_template | llm

# Retry wrapper for LLM pipeline with exponential backoff on rate limits and transient errors
@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def invoke_pipeline_with_retry(inputs: Dict):
    return pipeline.invoke(inputs)

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
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
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
        response_text = invoke_pipeline_with_retry({
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

@app.post("/escalate")
def escalate_to_human(current_user=Depends(get_current_user), db=Depends(get_db)):
    """
    Endpoint for escalation. Could integrate with CRM, Slack, etc.
    """
    # Find active conversation
    active_conversation = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.end_time.is_(None)
    ).order_by(desc(Conversation.start_time)).first()
    
    if not active_conversation:
        raise HTTPException(status_code=404, detail="No active conversation found")
    
    # Mark last AI message as escalated
    last_ai_message = db.query(Message).filter(
        Message.conversation_id == active_conversation.id,
        Message.role == "assistant"
    ).order_by(desc(Message.timestamp)).first()
    
    if last_ai_message:
        last_ai_message.was_escalated = True
        db.commit()
    
    return {"message": f"User {current_user.username} escalated to human support for conversation {active_conversation.id}."}

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

# ----- Admin Endpoints -----
@app.get("/admin/users", response_model=List[UserResponse])
def list_users(current_user: User = Depends(admin_required), db = Depends(get_db)):
    """
    List all users (admin only)
    """
    users = db.query(User).all()
    return users

@app.post("/admin/users/{user_id}/make-admin")
def make_admin(user_id: int, current_user: User = Depends(admin_required), db = Depends(get_db)):
    """
    Promote a user to admin (admin only)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_admin = True
    db.commit()
    
    return {"message": f"User {user.username} has been promoted to admin"}

# ----- Analytics Endpoints -----
@app.get("/admin/analytics", response_model=AnalyticsResponse)
def get_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(admin_required), 
    db = Depends(get_db)
):
    # Calculate time range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Total conversations in period
    total_conversations = db.query(func.count(Conversation.id)).filter(
        Conversation.start_time >= start_date
    ).scalar()
    
    # Total messages in period
    total_messages = db.query(func.count(Message.id)).join(Conversation).filter(
        Conversation.start_time >= start_date
    ).scalar()
    
    # Average messages per conversation
    avg_messages = float(total_messages / total_conversations) if total_conversations > 0 else 0.0
    
    # Escalation rates per conversation
    escalated_messages = db.query(func.count(Message.id)).filter(
        Message.timestamp >= start_date,
        Message.was_escalated == True
    ).scalar()
    
    escalation_rate = float(escalated_messages / total_messages * 100) if total_messages > 0 else 0.0
    
    # Estimate token usage and cost
    # Assume an average of 500 tokens per user message and 750 tokens per AI response
    user_messages = db.query(func.count(Message.id)).filter(
        Message.timestamp >= start_date,
        Message.role == "user"
    ).scalar()
    
    ai_messages = db.query(func.count(Message.id)).filter(
        Message.timestamp >= start_date,
        Message.role == "assistant"
    ).scalar()
    
    # Estimate token usage - convert any potential numpy types to Python native types
    estimated_user_tokens = int(user_messages * 500)
    estimated_ai_tokens = int(ai_messages * 750)
    
    # Calculate estimated cost (gpt-3.5-turbo-instruct pricing)
    # Input: $0.0015/1K tokens, Output: $0.0020/1K tokens
    estimated_input_cost = float((estimated_user_tokens / 1000) * 0.0015)
    estimated_output_cost = float((estimated_ai_tokens / 1000) * 0.0020)
    estimated_total_cost = float(estimated_input_cost + estimated_output_cost)
    
    # Top user questions (most frequent)
    top_questions_query = db.query(
        Message.content, 
        func.count(Message.id).label('count')
    ).filter(
        Message.timestamp >= start_date,
        Message.role == "user"
    ).group_by(Message.content).order_by(desc('count')).limit(10).all()
    
    top_questions = [{"question": q.content, "count": int(q.count)} for q in top_questions_query]
    
    # Daily usage
    daily_usage = []
    for day_offset in range(min(days, 30)):  # Limit to 30 points on chart
        day_date = end_date - timedelta(days=day_offset)
        day_start = day_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_count = db.query(func.count(Message.id)).filter(
            Message.timestamp >= day_start,
            Message.timestamp < day_end,
            Message.role == "user"
        ).scalar()
        
        daily_usage.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "message_count": int(day_count)
        })
    
    # Reverse to get chronological order
    daily_usage.reverse()
    
    return {
        "total_conversations": int(total_conversations),
        "total_messages": int(total_messages),
        "avg_messages_per_conversation": round(float(avg_messages), 2),
        "escalation_rate": round(float(escalation_rate), 2),
        "top_questions": top_questions,
        "daily_usage": daily_usage,
        "estimated_input_cost": round(float(estimated_input_cost), 4),
        "estimated_output_cost": round(float(estimated_output_cost), 4),
        "estimated_total_cost": round(float(estimated_total_cost), 4)
    }

@app.get("/conversation/history")
def get_conversation_history(current_user=Depends(get_current_user), db=Depends(get_db)):
    """
    Get the active conversation history for the current user
    """
    try:
        # Find the active conversation
        active_conversation = db.query(Conversation).filter(
            Conversation.user_id == current_user.id,
            Conversation.end_time.is_(None)
        ).order_by(desc(Conversation.start_time)).first()
        
        if not active_conversation:
            # No active conversation found
            return {"active_conversation_id": None, "messages": []}
        
        # Retrieve conversation messages
        messages = db.query(Message).filter(
            Message.conversation_id == active_conversation.id
        ).order_by(Message.timestamp).all()
        
        # Format messages for the response
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "confidence_score": float(msg.confidence_score) if msg.confidence_score else None,
                "was_escalated": msg.was_escalated
            })
        
        return {
            "active_conversation_id": active_conversation.id,
            "start_time": active_conversation.start_time.isoformat(),
            "messages": formatted_messages
        }
    
    except Exception as e:
        print(f"Error retrieving conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# For development use only - to be removed in production
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
