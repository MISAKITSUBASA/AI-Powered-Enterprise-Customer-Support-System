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

# Alternative import approach
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import Base, User, Conversation, Message, Document

# Import knowledge base
from knowledge_base import KnowledgeBase

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
if not OPENAI_API_KEY:
    raise ValueError("Please provide OPENAI_API_KEY in your .env file.")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    question: str

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{user_question}")
])


llm = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini"
)

chain = template | llm

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running"}

@app.post("/ask")
def ask_question(query: UserQuery):
    user_question = query.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    # Invoke the RunnableSequence with the user_question
    response = chain.invoke({"user_question": user_question})
    return {"answer": response}
