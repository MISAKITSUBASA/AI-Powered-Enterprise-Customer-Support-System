from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()
__all__ = ['Base', 'User', 'Conversation', 'Message', 'Document']

class User(Base):
    """User model for authentication and user management"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    email = Column(String, unique=True, index=True, nullable=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    documents = relationship("Document", back_populates="uploader")

class Conversation(Base):
    """Conversation model to track support sessions"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)  # Null when conversation is active
    channel = Column(String, default="web")  # Web, mobile, etc.
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    """Message model to store conversation messages"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # user, assistant, or system
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # AI-specific fields
    confidence_score = Column(Float, nullable=True)  # AI confidence in answer
    was_escalated = Column(Boolean, default=False)  # Escalated to human support
    used_kb = Column(Boolean, default=False)  # Used knowledge base for response
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class Document(Base):
    """Document model for knowledge base management"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    file_type = Column(String)  # pdf, txt, docx, etc.
    file_size = Column(Integer)  # Size in bytes
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    uploader_id = Column(Integer, ForeignKey("users.id"))
    vector_index_id = Column(Integer, nullable=True)  # Reference to FAISS index
    embedding_status = Column(String, default="pending")  # pending, completed, failed

    # Relationships
    uploader = relationship("User", back_populates="documents")
