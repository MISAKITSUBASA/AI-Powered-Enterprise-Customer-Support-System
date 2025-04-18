import os
import jwt
import bcrypt
import redis
import hashlib
from typing import Optional, Dict, List, Tuple
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
from sqlalchemy import create_engine, Column, Integer, String, func, desc
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import Base, User, Conversation, Message, Document

from knowledge_base import KnowledgeBase
from env_utils import get_openai_api_key, print_masked_key

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("app")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = get_openai_api_key()
JWT_SECRET = os.getenv("JWT_SECRET", "changeme")
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", "86400"))

# Redis cache configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TTL = int(os.getenv("REDIS_CACHE_TTL", "86400"))

# Initialize Redis connection
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    logger.info(f"Redis connected at {REDIS_URL}")
except Exception as e:
    logger.warning(f"Redis connection failed: {str(e)}. Caching will be disabled.")
    redis_client = None

def get_cache_key(question: str) -> str:
    """Generate unique cache key using question hash"""
    return f"ai_response:{hashlib.md5(question.encode()).hexdigest()}"

def get_cached_response(question: str) -> Optional[Dict]:
    """Retrieve cached response from Redis"""
    if not redis_client:
        return None
        
    cache_key = get_cache_key(question)
    cached = redis_client.get(cache_key)
    
    if cached:
        try:
            logger.info(f"Cache hit for question: {question[:30]}...")
            return json.loads(cached)
        except:
            return None
    return None

def cache_response(question: str, answer: str, confidence: float, escalate: bool, ttl: int = REDIS_TTL):
    """Cache high-confidence responses to Redis"""
    if not redis_client or confidence < 70:
        return False
        
    cache_key = get_cache_key(question)
    data = {
        "answer": answer,
        "confidence": confidence,
        "escalate": escalate,
        "cached_at": datetime.utcnow().isoformat()
    }
    
    try:
        redis_client.setex(cache_key, ttl, json.dumps(data))
        logger.info(f"Cached response for '{question[:30]}...' with confidence {confidence:.2f}")
        return True
    except Exception as e:
        logger.error(f"Error caching response: {str(e)}")
        return False

# Verify OpenAI API key
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")
else:
    print_masked_key(OPENAI_API_KEY, "Using OpenAI API Key")

# Initialize FastAPI app
app = FastAPI(title="AI Customer Support API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure database connection
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./customer_support.db")

if (DATABASE_URL.startswith("postgres://")):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create database tables
Base.metadata.create_all(bind=engine)

# Setup JWT authentication
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

KNOWLEDGE_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_index.index")

kb = KnowledgeBase(index_path=KNOWLEDGE_INDEX_PATH)

prompt_template = PromptTemplate(
    input_variables=["history", "user_input", "knowledge_base", "emotion_style"],
    template=(
        "You are an AI customer support assistant. Be helpful, clear, and professional.\n"
        "If you're unsure about an answer, be honest and suggest escalating to human support.\n\n"
        "The user's emotional tone appears to be: {emotion_style}. Adjust your response accordingly.\n\n"
        "Below is information from our knowledge base that might help answer the question:\n"
        "{knowledge_base}\n\n"
        "Conversation history:\n"
        "{history}\n"
        "User: {user_input}\n\n"
        "Provide your response in the following format:\n"
        "CONFIDENCE: [Rate your confidence in your answer from 0 to 100, where 100 means you're completely confident]\n"
        "ANSWER: [Your helpful response to the user]\n\n"
        "Only I can see the confidence rating, it won't be shown to the user."
    )
)

llm = OpenAI(
    openai_api_key=OPENAI_API_KEY, 
    temperature=0.5,
    model_name="gpt-3.5-turbo-instruct"
)
pipeline = prompt_template | llm

# Emotion analysis constants
EMOTION_CATEGORIES = {
    "angry": {
        "keywords": ["angry", "frustrated", "furious", "irritated", "upset", "annoyed"],
        "response_style": "calm, understanding, and solution-oriented"
    },
    "sad": {
        "keywords": ["sad", "disappointed", "unhappy", "heartbroken", "depressed", "down"],
        "response_style": "empathetic, supportive, and reassuring"
    },
    "anxious": {
        "keywords": ["anxious", "worried", "nervous", "stressed", "afraid", "concerned"],
        "response_style": "reassuring, clear, and thorough"
    },
    "confused": {
        "keywords": ["confused", "unsure", "uncertain", "puzzled", "lost", "unclear"],
        "response_style": "clear, step-by-step, and patient"
    },
    "urgent": {
        "keywords": ["urgent", "immediately", "asap", "emergency", "critical", "deadline"],
        "response_style": "prompt, direct, and solution-focused"
    },
    "positive": {
        "keywords": ["happy", "excited", "pleased", "grateful", "satisfied", "appreciate"],
        "response_style": "friendly, enthusiastic, and approachable"
    },
    "neutral": {
        "keywords": [],
        "response_style": "professional, informative, and balanced"
    }
}

def analyze_emotion(text: str) -> Tuple[str, float]:
    """
    Analyze the emotional tone of a text message.
    Returns the detected emotion category and confidence score.
    """
    # Basic rule-based emotion analysis
    text_lower = text.lower()
    
    # Default to neutral if we can't determine emotion
    detected_emotion = "neutral"
    max_score = 0.0
    
    # Simple keyword matching for emotion detection
    for emotion, data in EMOTION_CATEGORIES.items():
        if emotion == "neutral":
            continue
            
        score = 0.0
        for keyword in data["keywords"]:
            if keyword in text_lower:
                score += 1.0
                
        # Normalize score based on number of keywords
        if data["keywords"]:
            score = score / len(data["keywords"])
            
        if score > max_score:
            max_score = score
            detected_emotion = emotion
    
    # If we have access to OpenAI API, we can use it for more nuanced analysis
    # This is a more advanced implementation
    try:
        if max_score < 0.3 and OPENAI_API_KEY:  # If basic detection isn't confident enough
            prompt = f"""
            Analyze the emotional tone of the following customer message:
            
            "{text}"
            
            Categorize the emotion as one of: angry, sad, anxious, confused, urgent, positive, or neutral.
            Respond with only the emotion category and confidence score (0-1) separated by a comma.
            For example: "angry,0.8" or "neutral,0.6"
            """
            
            response = OpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0
            ).invoke(prompt)
            
            try:
                emotion, confidence = response.strip().split(",")
                confidence = float(confidence)
                if confidence > max_score and emotion in EMOTION_CATEGORIES:
                    detected_emotion = emotion
                    max_score = confidence
            except:
                # Fall back to rule-based result if parsing fails
                logger.warning(f"Failed to parse OpenAI emotion analysis response: {response}")
                pass
    except Exception as e:
        logger.warning(f"OpenAI emotion analysis failed: {str(e)}")
        # Continue with rule-based result
    
    return detected_emotion, max_score

@app.get("/")
def root():
    return {"message": "AI-Powered Enterprise Customer Support System API", 
            "version": "1.0.0",
            "docs": "/docs"}

@app.post("/register")
def register(user_data: UserCreate, db=Depends(get_db)):
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    if user_data.email:
        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt())
    
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
    
    if not bcrypt.checkpw(user_data.password.encode("utf-8"), user.password_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
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
    Multi-turn chat endpoint with knowledge base integration and emotion analysis
    """
    try:
        user_id = current_user.id
        user_input = request.question.strip()
        using_cache = False
        
        if not user_input:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Track request count in Redis for statistics
        if redis_client:
            redis_client.incr("cache_request_count")

        # Check Redis cache first for high-confidence responses
        cached_response = get_cached_response(user_input)
        
        if cached_response:
            # Track cache hit for statistics
            if redis_client:
                redis_client.incr("cache_hit_count")
                
            logger.info(f"Cache hit for question: {user_input[:30]}...")
            response_text = cached_response["answer"]
            confidence_score = cached_response["confidence"]
            escalate = cached_response["escalate"]
            using_cache = True
            # For cached responses, we don't have emotion information
            detected_emotion = "neutral"
            emotion_confidence = 0.0
        else:
            # Not in cache, analyze emotion and generate response
            logger.info(f"Cache miss for question: {user_input[:30]}...")
            
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
                logger.error(f"Knowledge base search error: {str(e)}")
                knowledge_results = []
                
            # Format knowledge base results
            knowledge_base_text = "\n\n".join([
                f"Document: {result['metadata'].get('file_name', 'Unknown')}\n"
                f"Content: {result['content']}\n"
                f"Relevance: {result['score']:.2f}"
                for result in knowledge_results
            ]) if knowledge_results else "No relevant information found in knowledge base."
            
            # Analyze user's emotional state
            detected_emotion, emotion_confidence = analyze_emotion(user_input)
            emotion_style = EMOTION_CATEGORIES[detected_emotion]["response_style"]
            
            # Log detected emotion
            logger.info(f"Detected emotion: {detected_emotion} (confidence: {emotion_confidence:.2f})")
            
            # Invoke LLM pipeline with emotion context
            full_response = pipeline.invoke({
                "history": history_formatted,
                "user_input": user_input,
                "knowledge_base": knowledge_base_text,
                "emotion_style": emotion_style
            })
            
            # Log the raw response from OpenAI
            logger.info(f"Raw OpenAI response: {full_response}")
            
            # Parse the response to extract confidence and answer
            response_text = ""
            ai_confidence = 0
            
            try:
                # Extract confidence level from response
                if "CONFIDENCE:" in full_response and "ANSWER:" in full_response:
                    confidence_line = full_response.split("CONFIDENCE:")[1].split("\n")[0].strip()
                    ai_confidence = float(confidence_line.split()[0])
                    
                    # Extract answer (everything after ANSWER:)
                    answer_part = full_response.split("ANSWER:")[1].strip()
                    response_text = answer_part
                else:
                    # Fallback if format is not followed
                    response_text = full_response
                    # Use knowledge base confidence as fallback
                    if knowledge_results:
                        weights = [0.6, 0.3, 0.1]
                        weighted_scores = [result["score"] * weight for result, weight in zip(knowledge_results, weights)]
                        ai_confidence = min(100, sum(weighted_scores) / sum(weights[:len(knowledge_results)]) * 100)
                    else:
                        ai_confidence = 50
            except Exception as e:
                logger.error(f"Error parsing AI response: {str(e)}")
                response_text = full_response
                ai_confidence = 50
                
            logger.info(f"Extracted confidence: {ai_confidence}, Answer length: {len(response_text)}")
            
            # Use the AI's own confidence score instead of calculated confidence
            confidence_score = ai_confidence
            
            # Determine if escalation is needed (confidence below threshold)
            escalate = confidence_score < 70
            
            # Cache high-confidence responses (≥70%)
            if confidence_score >= 70:
                cache_response(user_input, response_text, confidence_score, escalate)
        
        # Whether cached or not, we need to ensure there's an active conversation
        if using_cache:
            active_conversation = db.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.end_time.is_(None)
            ).order_by(desc(Conversation.start_time)).first()
            
            if not active_conversation:
                active_conversation = Conversation(user_id=user_id)
                db.add(active_conversation)
                db.commit()
                db.refresh(active_conversation)
        
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
            used_kb=not using_cache and len(knowledge_results) > 0 if 'knowledge_results' in locals() else False
        )
        db.add(ai_message)
        
        db.commit()
        
        # Return emotion data to frontend
        return {
            "answer": response_text,
            "confidence": confidence_score,
            "escalate": escalate,
            "conversation_id": active_conversation.id,
            "from_cache": using_cache,
            "emotion": {
                "category": detected_emotion if 'detected_emotion' in locals() else "neutral",
                "confidence": emotion_confidence if 'emotion_confidence' in locals() else 0.0
            }
        }
    except Exception as e:
        db.rollback()  # Rollback any pending transaction on error
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/escalate")
def escalate_to_human(current_user=Depends(get_current_user), db=Depends(get_db)):
    active_conversation = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.end_time.is_(None)
    ).order_by(desc(Conversation.start_time)).first()
    
    if not active_conversation:
        raise HTTPException(status_code=404, detail="No active conversation found")
    
    last_ai_message = db.query(Message).filter(
        Message.conversation_id == active_conversation.id,
        Message.role == "assistant"
    ).order_by(desc(Message.timestamp)).first()
    
    if last_ai_message:
        last_ai_message.was_escalated = True
        db.commit()
    
    return {"message": f"User {current_user.username} escalated to human support for conversation {active_conversation.id}."}

@app.post("/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...), 
    current_user: User = Depends(admin_required),
    db = Depends(get_db)
):
    file_size = 0
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    file_name = file.filename
    file_extension = file_name.split(".")[-1].lower() if "." in file_name else ""
    
    supported_types = ["pdf", "txt", "docx", "doc", "xlsx", "xls"]
    if file_extension not in supported_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types: {', '.join(supported_types)}"
        )
    
    success = kb.add_document(
        file_content=file_content,
        file_name=file_name,
        file_type=file_extension,
        metadata={"uploader": current_user.username}
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process document")
    
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
    results = kb.search(query, top_k=top_k)
    
    formatted_results = []
    for result in results:
        formatted_results.append({
            "content": result["content"],
            "file_name": result["metadata"].get("file_name", "Unknown"),
            "relevance_score": result["score"]
        })
    
    return {"results": formatted_results}

@app.get("/admin/users", response_model=List[UserResponse])
def list_users(current_user: User = Depends(admin_required), db = Depends(get_db)):
    users = db.query(User).all()
    return users

@app.post("/admin/users/{user_id}/make-admin")
def make_admin(user_id: int, current_user: User = Depends(admin_required), db = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_admin = True
    db.commit()
    
    return {"message": f"User {user.username} has been promoted to admin"}

@app.get("/admin/analytics", response_model=AnalyticsResponse)
def get_analytics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(admin_required), 
    db = Depends(get_db)
):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    total_conversations = db.query(func.count(Conversation.id)).filter(
        Conversation.start_time >= start_date
    ).scalar()
    
    total_messages = db.query(func.count(Message.id)).join(Conversation).filter(
        Conversation.start_time >= start_date
    ).scalar()
    
    avg_messages = float(total_messages / total_conversations) if total_conversations > 0 else 0.0
    
    escalated_messages = db.query(func.count(Message.id)).filter(
        Message.timestamp >= start_date,
        Message.was_escalated == True
    ).scalar()
    
    escalation_rate = float(escalated_messages / total_messages * 100) if total_messages > 0 else 0.0
    
    user_messages = db.query(func.count(Message.id)).filter(
        Message.timestamp >= start_date,
        Message.role == "user"
    ).scalar()
    
    ai_messages = db.query(func.count(Message.id)).filter(
        Message.timestamp >= start_date,
        Message.role == "assistant"
    ).scalar()
    
    estimated_user_tokens = int(user_messages * 500)
    estimated_ai_tokens = int(ai_messages * 750)
    
    estimated_input_cost = float((estimated_user_tokens / 1000) * 0.0015)
    estimated_output_cost = float((estimated_ai_tokens / 1000) * 0.0020)
    estimated_total_cost = float(estimated_input_cost + estimated_output_cost)
    
    top_questions_query = db.query(
        Message.content, 
        func.count(Message.id).label('count')
    ).filter(
        Message.timestamp >= start_date,
        Message.role == "user"
    ).group_by(Message.content).order_by(desc('count')).limit(10).all()
    
    top_questions = [{"question": q.content, "count": int(q.count)} for q in top_questions_query]
    
    daily_usage = []
    for day_offset in range(min(days, 30)):
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
    
    daily_usage.reverse()
    
    cache_statistics = None
    if redis_client:
        try:
            all_keys = redis_client.keys("ai_response:*")
            total_keys = len(all_keys)
            
            if total_keys > 0:
                hit_count = redis_client.get("cache_hit_count")
                hit_count = int(hit_count) if hit_count else 0
                
                request_count = redis_client.get("cache_request_count") 
                request_count = int(request_count) if request_count else user_messages
                
                hit_rate = (hit_count / request_count * 100) if request_count > 0 else 0
                
                confidence_sum = 0
                for key in all_keys[:100]:
                    data = redis_client.get(key)
                    if data:
                        try:
                            data_json = json.loads(data)
                            confidence_sum += data_json.get("confidence", 0)
                        except:
                            pass
                
                avg_confidence = confidence_sum / min(len(all_keys), 100) if all_keys else 0
                
                input_savings = (hit_count * 500 / 1000) * 0.0015
                output_savings = (hit_count * 750 / 1000) * 0.0020
                total_savings = input_savings + output_savings
                
                cache_statistics = {
                    "total_requests": request_count,
                    "hits": hit_count,
                    "hit_rate": hit_rate,
                    "avg_confidence": avg_confidence,
                    "estimated_savings": round(float(total_savings), 4)
                }
        except Exception as e:
            logger.error(f"Error calculating cache statistics: {str(e)}")
    
    response = {
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
    
    if cache_statistics:
        response["cache_statistics"] = cache_statistics
    
    return response

@app.get("/conversation/history")
def get_conversation_history(current_user=Depends(get_current_user), db=Depends(get_db)):
    try:
        active_conversation = db.query(Conversation).filter(
            Conversation.user_id == current_user.id,
            Conversation.end_time.is_(None)
        ).order_by(desc(Conversation.start_time)).first()
        
        if not active_conversation:
            return {"active_conversation_id": None, "messages": []}
        
        messages = db.query(Message).filter(
            Message.conversation_id == active_conversation.id
        ).order_by(Message.timestamp).all()
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
